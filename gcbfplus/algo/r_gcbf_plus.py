import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import einops as ei
import os
import pickle

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
from .gcbf_plus import GCBFPlus
from .gcbf_plus import Batch
from .module.adversary import AdversarialDynamicsNoise


class RGCBFPlus(GCBFPlus):

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
            lr_noise: float = 1e-5,
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
        super(RGCBFPlus, self).__init__(
            env, node_dim, edge_dim, state_dim, action_dim, n_agents, gnn_layers, batch_size,
            buffer_size, horizon, lr_actor, lr_cbf, alpha, eps, inner_epoch, loss_action_coef,
            loss_unsafe_coef, loss_safe_coef, loss_h_dot_coef, max_grad_norm, seed, **kwargs
        )

        self.noise = AdversarialDynamicsNoise(
            n_dim=edge_dim,
            bounds=env.noise_bounds
        )
        noise_key, self.key = jr.split(self.key, 2)
        noise_params = self.noise.init(noise_key, self.nominal_graph, n_agents)
        noise_optim = optax.adamw(learning_rate=lr_noise, weight_decay=1e-3)
        self.noise_optim = optax.apply_if_finite(noise_optim, 1_000_000)
        self.noise_train_state = TrainState.create(
            apply_fn=self.noise.get,
            params=noise_params,
            tx=self.noise_optim,
        )

    def update_nets(self, rollout: Rollout, safe_mask, unsafe_mask):
        update_info = {}

        # Compute b_u_qp.
        n_chunks = 8
        batch_size = len(rollout.graph.states)
        chunk_size = batch_size // n_chunks

        # t0 = time.time()
        b_u_qp = []
        for ii in range(n_chunks):
            graph = jtu.tree_map(lambda x: x[ii * chunk_size: (ii + 1) * chunk_size], rollout.graph)
            b_u_qp.append(jax2np(self.get_b_u_qp(graph, self.cbf_tgt.params)))
        b_u_qp = tree_merge(b_u_qp)

        batch_orig = Batch(rollout.graph, safe_mask, unsafe_mask, b_u_qp)

        for i_epoch in range(self.inner_epoch):
            idx = self.rng.choice(rollout.length, size=rollout.length, replace=False)
            # (n_mb, mb_size)
            batch_idx = np.stack(np.array_split(idx, idx.shape[0] // self.batch_size), axis=0)
            batch = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            cbf_train_state, actor_train_state, noise_train_state, update_info = self.update_inner(
                self.cbf_train_state, self.actor_train_state, self.noise_train_state, batch
            )
            self.cbf_train_state = cbf_train_state
            self.actor_train_state = actor_train_state
            self.noise_train_state = noise_train_state

        # Update target.
        self.cbf_tgt = self.update_tgt(self.cbf_tgt, self.cbf_train_state, 0.5)

        return update_info

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def update_inner(
            self,
            cbf_train_state: TrainState,
            actor_train_state: TrainState,
            noise_train_state: TrainState,
            batch: Batch
    ) -> tuple[TrainState, TrainState, TrainState, dict]:
        def update_fn(carry, minibatch: Batch):
            cbf, actor, noise = carry
            # (batch_size, n_agents) -> (minibatch_size * n_agents, )
            safe_mask_batch = merge01(minibatch.safe_mask)
            unsafe_mask_batch = merge01(minibatch.unsafe_mask)

            def get_loss(cbf_params: Params, actor_params: Params) -> Tuple[Array, dict]:
                # get CBF values
                cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf_params))
                cbf_fn_no_grad = jax_vmap(ft.partial(self.cbf.get_cbf, jax.lax.stop_gradient(cbf_params)))
                # (minibatch_size, n_agents)
                h = cbf_fn(minibatch.graph).squeeze()
                # (minibatch_size * n_agents,)
                h = merge01(h)

                # unsafe region h(x) < 0
                unsafe_data_ratio = jnp.mean(unsafe_mask_batch)
                h_unsafe = jnp.where(unsafe_mask_batch, h, -jnp.ones_like(h) * self.eps * 2)
                max_val_unsafe = jax.nn.relu(h_unsafe + self.eps)
                loss_unsafe = jnp.sum(max_val_unsafe) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)
                acc_unsafe_mask = jnp.where(unsafe_mask_batch, h, jnp.ones_like(h))
                acc_unsafe = (jnp.sum(jnp.less(acc_unsafe_mask, 0)) + 1e-6) / (
                            jnp.count_nonzero(unsafe_mask_batch) + 1e-6)

                # safe region h(x) > 0
                h_safe = jnp.where(safe_mask_batch, h, jnp.ones_like(h) * self.eps * 2)
                max_val_safe = jax.nn.relu(-h_safe + self.eps)
                loss_safe = jnp.sum(max_val_safe) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)
                acc_safe_mask = jnp.where(safe_mask_batch, h, -jnp.ones_like(h))
                acc_safe = (jnp.sum(jnp.greater(acc_safe_mask, 0)) + 1e-6) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)

                # get neural network actions
                action_fn = jax.vmap(ft.partial(self.act, params=actor_params))
                action = action_fn(minibatch.graph)

                # get next graph
                forward_fn = ft.partial(self._env.forward_graph,
                                        noise_model=ft.partial(self.noise_train_state.apply_fn,
                                                               noise.params, n_agents=self.n_agents))
                forward_fn = jax.vmap(forward_fn)
                next_graph = forward_fn(minibatch.graph, action)
                h_next = merge01(cbf_fn(next_graph).squeeze())
                h_dot = (h_next - h) / self._env.dt

                # stop gradient and get next graph
                h_no_grad = jax.lax.stop_gradient(h)
                h_next_no_grad = merge01(cbf_fn_no_grad(next_graph).squeeze())
                h_dot_no_grad = (h_next_no_grad - h_no_grad) / self._env.dt

                # h_dot + alpha * h > 0 (backpropagate to action, and backpropagate to h when labeled)
                labeled_mask = jnp.logical_or(unsafe_mask_batch, safe_mask_batch)
                max_val_h_dot = jax.nn.relu(-h_dot - self.alpha * h + self.eps)
                max_val_h_dot_no_grad = jax.nn.relu(-h_dot_no_grad - self.alpha * h + self.eps)
                max_val_h_dot = jnp.where(labeled_mask, max_val_h_dot, max_val_h_dot_no_grad)
                loss_h_dot = jnp.mean(max_val_h_dot)
                acc_h_dot = jnp.mean(jnp.greater(h_dot + self.alpha * h, 0))

                # action loss
                assert action.shape == minibatch.u_qp.shape
                loss_action = jnp.mean(jnp.square(action - minibatch.u_qp).sum(axis=-1))

                # total loss
                total_loss = (
                        self.loss_action_coef * loss_action
                        + self.loss_unsafe_coef * loss_unsafe
                        + self.loss_safe_coef * loss_safe
                        + self.loss_h_dot_coef * loss_h_dot
                )

                return total_loss, {'loss/action': loss_action,
                                    'loss/unsafe': loss_unsafe,
                                    'loss/safe': loss_safe,
                                    'loss/h_dot': loss_h_dot,
                                    'loss/total': total_loss,
                                    'acc/unsafe': acc_unsafe,
                                    'acc/safe': acc_safe,
                                    'acc/h_dot': acc_h_dot,
                                    'acc/unsafe_data_ratio': unsafe_data_ratio}

            def get_noise_loss(noise_params: Params) -> [Array, dict]:
                # get CBF values
                cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf.params))
                # (minibatch_size, n_agents)
                h = cbf_fn(minibatch.graph).squeeze()
                # (minibatch_size * n_agents,)
                h = merge01(h)

                # get neural network actions
                action_fn = jax.vmap(ft.partial(self.act, params=actor.params))
                action = action_fn(minibatch.graph)

                # get next graph
                forward_fn = ft.partial(self._env.forward_graph,
                                        noise_model=ft.partial(self.noise_train_state.apply_fn,
                                                               noise_params, n_agents=self.n_agents))
                forward_fn = jax.vmap(forward_fn)
                next_graph = forward_fn(minibatch.graph, action)
                h_next = merge01(cbf_fn(next_graph).squeeze())
                h_dot = (h_next - h) / self._env.dt
                loss_noise = -jax.nn.relu(-h_dot - self.alpha * h + self.eps * 10)

                noise_fn = ft.partial(self.noise_train_state.apply_fn,
                                                               noise_params, n_agents=self.n_agents)
                noise_val = jax.vmap(noise_fn)(minibatch.graph)
                # noise_val = self.noise_train_state.apply_fn(noise_params, minibatch.graph, self.n_agents)
                loss_noise += jnp.mean(jnp.square(noise_val)) * 0.01

                return jnp.mean(loss_noise), {'loss/noise': jnp.mean(loss_noise)}

            (loss, loss_info), (grad_cbf, grad_actor) = jax.value_and_grad(
                get_loss, has_aux=True, argnums=(0, 1))(cbf.params, actor.params)
            (loss_noise, loss_noise_info), grad_noise = jax.value_and_grad(
                get_noise_loss, has_aux=True)(noise.params)
            grad_cbf, grad_cbf_norm = compute_norm_and_clip(grad_cbf, self.max_grad_norm)
            grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
            grad_noise, grad_noise_norm = compute_norm_and_clip(grad_noise, self.max_grad_norm)
            cbf = cbf.apply_gradients(grads=grad_cbf)
            actor = actor.apply_gradients(grads=grad_actor)
            noise = noise.apply_gradients(grads=grad_noise)
            grad_info = {'grad_norm/cbf': grad_cbf_norm,
                         'grad_norm/actor': grad_actor_norm,
                         'grad_norm/noise': grad_noise_norm}
            return (cbf, actor, noise), grad_info | loss_info | loss_noise_info

        train_state = (cbf_train_state, actor_train_state, noise_train_state)
        (cbf_train_state, actor_train_state, noise_train_state), info = lax.scan(update_fn, train_state, batch)

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)
        return cbf_train_state, actor_train_state, noise_train_state, info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.actor_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.cbf_train_state.params, open(os.path.join(model_dir, 'cbf.pkl'), 'wb'))
        pickle.dump(self.noise_train_state.params, open(os.path.join(model_dir, 'noise.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.actor_train_state = \
            self.actor_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.cbf_train_state = \
            self.cbf_train_state.replace(params=pickle.load(open(os.path.join(path, 'cbf.pkl'), 'rb')))
        self.noise_train_state = \
            self.noise_train_state.replace(params=pickle.load(open(os.path.join(path, 'noise.pkl'), 'rb')))
