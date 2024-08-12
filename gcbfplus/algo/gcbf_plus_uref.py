import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import einops as ei

from typing import Optional, Tuple, NamedTuple
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array, State
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import merge01, jax_vmap, mask2index, tree_merge
from gcbfplus.trainer.data import Rollout
from gcbfplus.trainer.buffer import MaskedReplayBuffer
from gcbfplus.trainer.utils import compute_norm_and_clip, jax2np, tree_copy, empty_grad_tx
from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.algo.module.cbf import CBF
from gcbfplus.algo.module.policy import DeterministicPolicy
from .gcbf import GCBF
from .gcbf_plus import GCBFPlus


class Batch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_qp: Action


class GCBFPlusUref(GCBFPlus):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            gnn_layers: int,
            batch_size: int,
            buffer_size: int,
            horizon: int = 32,
            lr_actor: float = 3e-5,
            lr_cbf: float = 3e-5,
            alpha: float = 1.0,
            eps: float = 0.02,
            inner_epoch: int = 8,
            loss_action_coef: float = 0.001,
            loss_unsafe_coef: float = 1.,
            loss_safe_coef: float = 1.,
            loss_h_dot_coef: float = 0.2,
            max_grad_norm: float = 2.,
            seed: int = 0,
            **kwargs
    ):
        super(GCBFPlusUref, self).__init__(
            env, node_dim, edge_dim, state_dim, action_dim, n_agents, gnn_layers, batch_size, buffer_size,
            horizon, lr_actor, lr_cbf, alpha, eps, inner_epoch, loss_action_coef, loss_unsafe_coef, loss_safe_coef,
            loss_h_dot_coef, max_grad_norm, seed, **kwargs
        )

        # actor
        self.actor = DeterministicPolicy(node_dim, edge_dim, n_agents, action_dim, gnn_layers, ref_input=True)
        actor_key, self.key = jr.split(self.key)
        nominal_uref = jnp.zeros((n_agents, action_dim))
        actor_params = self.actor.net.init(actor_key, self.nominal_graph, self.n_agents, nominal_uref)
        actor_optim = optax.adamw(learning_rate=lr_actor, weight_decay=1e-3)
        self.actor_optim = optax.apply_if_finite(actor_optim, 1_000_000)
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.sample_action,
            params=actor_params,
            tx=self.actor_optim
        )

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_train_state.params
        u_ref = self._env.u_ref(graph)
        action = 2 * self.actor.get_action(params, graph, u_ref) + u_ref
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        u_ref = self._env.u_ref(graph)
        action, log_pi = self.actor_train_state.apply_fn(params, graph, key, u_ref)
        return 2 * action + u_ref, log_pi

