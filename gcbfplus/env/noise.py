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


class Noise:

    def __init__(self, n_agent: int, n_dim: int, stds: Array, bounds: Array):
        self._n_agent = n_agent
        self._n_dim = n_dim
        self._stds = stds
        self._bounds = bounds

    @property
    def bounds(self):
        return self._bounds

    def sample(self, params: Params | None, graph: GraphsTuple, key: PRNGKey) -> GraphsTuple:
        """
        Given a graph, return a new graph with noise added to the edge features.
        """
        noise_shape = graph.edges.shape
        noise = jr.uniform(key, noise_shape, minval=-self._bounds, maxval=self._bounds)
        # noise = jr.normal(key, noise_shape) * self._stds
        # noise = noise.clip(-self._bounds, self._bounds)

        # no noise on the agent-goal edges
        edge_is_goal = (self._n_agent <= graph.senders) & (graph.senders < self._n_agent * 2)
        noise = jnp.where(edge_is_goal[:, None], jnp.zeros_like(noise), noise)

        return graph._replace(edges=graph.edges + noise)


class NoiseNet(nn.Module):
    base_cls: Type[GNN]
    # head_cls: Type[nn.Module]
    edge_dim: int

    @nn.compact
    def __call__(self, obs: GraphsTuple) -> EdgeAttr:
        x = self.base_cls()(obs)
        node_feats_send = jtu.tree_map(lambda n: safe_get(n, obs.senders), x)
        node_feats_recv = jtu.tree_map(lambda n: safe_get(n, obs.receivers), x)
        edge_feats = jnp.concatenate([node_feats_recv, node_feats_send], axis=-1)
        edge_feats = nn.tanh(nn.Dense(self.edge_dim, kernel_init=default_nn_init())(edge_feats))
        return edge_feats


class AdversarialNoise(Noise):

    def __init__(self, n_agent: int, n_dim: int, stds: Array, bounds: Array):
        super().__init__(n_agent, n_dim, stds, bounds)
        self.gnn = GNN(
            msg_dim=16,
            hid_size_msg=(32, 32),
            hid_size_aggr=(16, 16),
            hid_size_update=(32, 32),
            out_dim=16,
            n_layers=1
        )
        self.noise_net = NoiseNet(base_cls=self.gnn, edge_dim=n_dim)

    def init(self, graph: GraphsTuple, key: PRNGKey) -> Params:
        return self.noise_net.init(key, graph)

    def sample(self, params: Params, graph: GraphsTuple, key: PRNGKey) -> GraphsTuple:
        edges_noise = self.noise_net.apply(params, graph)
        edges_noise = edges_noise * self.bounds
        return graph._replace(edges=graph.edges + edges_noise)


