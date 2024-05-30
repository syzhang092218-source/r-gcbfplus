import jax.random as jr
import jax.numpy as jnp
import flax.linen as nn
import jax.tree_util as jtu
import functools as ft

from typing import Type

from gcbfplus.nn.mlp import MLP

from gcbfplus.nn.utils import safe_get, default_nn_init
from gcbfplus.nn.gnn import GNN
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.typing import PRNGKey, Array, Params, EdgeAttr


class BoundedNodeNoise(nn.Module):
    base_cls: Type[GNN]
    bounds: Array  # shape: (n_dim,)

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Array:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        x = nn.tanh(nn.Dense(self.bounds.shape[0], kernel_init=default_nn_init())(x))
        return x * self.bounds


class AdversarialDynamicsNoise:

    def __init__(self, n_dim: int, bounds: Array):
        self._n_dim = n_dim
        self._bounds = bounds
        self.gnn = ft.partial(
            GNN,
            msg_dim=16,
            hid_size_msg=(32, 32),
            hid_size_aggr=(16, 16),
            hid_size_update=(32, 32),
            out_dim=16,
            n_layers=1
        )
        self.noise_net = BoundedNodeNoise(
            base_cls=self.gnn,
            bounds=self._bounds
        )

    def init(self, key: PRNGKey, graph: GraphsTuple, n_agents: int) -> Params:
        return self.noise_net.init(key, graph, n_agents)

    def get(self, params: Params, graph: GraphsTuple, n_agents: int) -> Array:
        return self.noise_net.apply(params, graph, n_agents)
