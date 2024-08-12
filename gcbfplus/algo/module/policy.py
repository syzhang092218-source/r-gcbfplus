import flax.linen as nn
import functools as ft
import numpy as np
import jax.nn as jnn
import jax.numpy as jnp
import jax

from typing import Type, Tuple
from abc import ABC, abstractproperty, abstractmethod

from .distribution import TanhTransformedDistribution, tfd
from ...utils.typing import Action, Array
from ...utils.graph import GraphsTuple
from ...nn.utils import default_nn_init, scaled_init
from ...nn.gnn import GNN
from ...nn.mlp import MLP
from ...utils.typing import PRNGKey, Params


class PolicyDistribution(nn.Module, ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tfd.Distribution:
        pass

    @abstractproperty
    def nu(self) -> int:
        pass


class TanhNormal(PolicyDistribution):
    base_cls: Type[GNN]
    _nu: int
    scale_final: float = 0.01
    std_dev_min: float = 1e-5
    std_dev_init: float = 0.5

    @property
    def std_dev_init_inv(self):
        # inverse of log(sum(exp())).
        inv = np.log(np.exp(self.std_dev_init) - 1)
        assert np.allclose(np.logaddexp(inv, 0), self.std_dev_init)
        return inv

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        # x = x.nodes
        scaler_init = scaled_init(default_nn_init(), self.scale_final)
        feats_scaled = nn.Dense(256, kernel_init=scaler_init, name="ScaleHid")(x)

        means = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseMean")(feats_scaled)
        stds_trans = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseStdTrans")(feats_scaled)
        stds = jnn.softplus(stds_trans + self.std_dev_init_inv) + self.std_dev_min

        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)

    @property
    def nu(self):
        return self._nu


class Deterministic(nn.Module):
    base_cls: Type[GNN]
    head_cls: Type[nn.Module]
    _nu: int

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, u_ref: Action = None, *args, **kwargs) -> Action:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if u_ref is not None:
            u_ref_encode = nn.tanh(nn.Dense(128, kernel_init=default_nn_init(), name="RefDense")(u_ref))
            x = jnp.concatenate([x, u_ref_encode], axis=-1)
        x = self.head_cls()(x)
        x = nn.tanh(nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDense")(x))
        return x


# class DeterministicWithRef(nn.Module):
#     base_cls: Type[GNN]
#     head_cls: Type[nn.Module]
#     action_encoder_cls: Type[nn.Module]
#     _nu: int
#
#     @nn.compact
#     def __call__(self, obs: GraphsTuple, n_agents: int, u_ref: Action, *args, **kwargs) -> Action:
#         x = self.base_cls()(obs, node_type=0, n_type=n_agents)
#         y = self.action_encoder_cls()(u_ref)
#         x = jnp.concatenate([x, y], axis=-1)
#         x = self.head_cls()(x)
#         x = nn.tanh(nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDense")(x))
#         return x


class MultiAgentPolicy(ABC):

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, action_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    @abstractmethod
    def get_action(self, params: Params, obs: GraphsTuple, u_ref: Action = None) -> Action:
        pass

    @abstractmethod
    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey, u_ref: Action = None) -> Tuple[Action, Array]:
        pass

    @abstractmethod
    def eval_action(
            self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey, u_ref: Action = None
    ) -> Tuple[Array, Array]:
        pass


class DeterministicPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            gnn_layers: int = 1,
            ref_input: bool = False
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.policy_base = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='PolicyHead'
        )
        self.ref_input = ref_input
        # if self.ref_input:
        #     self.net = DeterministicWithRef(
        #         base_cls=self.policy_base,
        #         head_cls=self.policy_head,
        #         _nu=action_dim
        #     )
        # else:
        self.net = Deterministic(base_cls=self.policy_base, head_cls=self.policy_head, _nu=action_dim)
        # self.std = 0.1

    def get_action(self, params: Params, obs: GraphsTuple, u_ref: Action = None) -> Action:
        return self.net.apply(params, obs, self.n_agents, u_ref)

    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey, u_ref: Action = None) -> Tuple[Action, Array]:
        action = self.get_action(params, obs, u_ref)
        log_pi = jnp.zeros_like(action)
        return action, log_pi

    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey, u_ref: Action = None) -> Tuple[Array, Array]:
        raise NotImplementedError


class PPOPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            gnn_layers: int = 1,
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.dist_base = ft.partial(
            GNN,
            msg_dim=64,
            hid_size_msg=(128, 128),
            hid_size_aggr=(128, 128),
            hid_size_update=(128, 128),
            out_dim=64,
            n_layers=gnn_layers
        )
        self.dist = TanhNormal(base_cls=self.dist_base, _nu=action_dim)

    def get_action(self, params: Params, obs: GraphsTuple, u_ref: Action = None) -> Action:
        dist = self.dist.apply(params, obs, n_agents=self.n_agents, u_ref=u_ref)
        action = dist.mode()
        return action

    def sample_action(
            self, params: Params, obs: GraphsTuple, key: PRNGKey, u_ref: Action = None
    ) -> Tuple[Action, Array]:
        dist = self.dist.apply(params, obs, n_agents=self.n_agents, u_ref=u_ref)
        action = dist.sample(seed=key)
        log_pi = dist.log_prob(action)
        return action, log_pi

    def eval_action(
            self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey, u_ref: Action = None
    ) -> Tuple[Array, Array]:
        dist = self.dist.apply(params, obs, n_agents=self.n_agents, u_ref=u_ref)
        log_pi = dist.log_prob(action)
        entropy = dist.entropy(seed=key)
        return log_pi, entropy
