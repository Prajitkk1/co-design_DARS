"""
Author: Steve Paul 
Date: 1/18/22 """
import torch

from stable_baselines3.common.policies import BasePolicy
import torch as th
import gym
import math
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)
from typing import NamedTuple
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from torch.nn import functional as F

#   TODO:
#   Make the policy network task independent

def dicttuple(cls: tuple):
    """Extends a tuple class with methods for the dict constructor."""

    cls.keys = lambda self: self._fields
    cls.__getitem__ = _getitem
    return cls

def _getitem(instance, index_or_key):
    """Returns the respective item."""

    if isinstance(index_or_key, str):
        try:
            return getattr(instance, index_or_key)
        except AttributeError:
            raise IndexError(index_or_key) from None

    return super().__getitem__(index_or_key)

@dicttuple
class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: th.Tensor
    context_node_projected: th.Tensor
    glimpse_key: th.Tensor
    glimpse_val: th.Tensor
    logit_key: th.Tensor

    # def __new__(cls, name, bases, namespace):
    #     my_fancy_new_namespace = {'__module__': module}
    #     if '__classcell__' in namespace:
    #         my_fancy_new_namespace['__classcell__'] = namespace['__classcell__']
    #     return super().__new__(cls, name, bases, my_fancy_new_namespace)


class ActorCriticGCAPSPolicy(BasePolicy):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[th.nn.Module] = th.nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: Union[th.device, str] = "cpu"
                 ):
        super(ActorCriticGCAPSPolicy, self).__init__(observation_space,
                                                     action_space,
                                                     features_extractor_class,
                                                     features_extractor_kwargs,
                                                     optimizer_class=optimizer_class,
                                                     optimizer_kwargs=optimizer_kwargs,
                                                     squash_output=squash_output)

        features_dim = features_extractor_kwargs['features_dim']
        node_dim = features_extractor_kwargs['node_dim']
        agent_node_dim = features_extractor_kwargs['agent_node_dim']
        self.node_dim = features_extractor_kwargs['node_dim']

        value_net_net = [th.nn.Linear(features_dim, features_dim, bias=True),
                         th.nn.Linear(features_dim, 1, bias=True)]
        self.value_net = th.nn.Sequential(*value_net_net).to(device=device)
        if features_extractor_kwargs['feature_extractor'] == "CAPAM":
            from Feature_Extractors import CAPAM
            self.features_extractor = CAPAM(
                node_dim=node_dim,
                features_dim=features_dim,
                K=features_extractor_kwargs['K'],
                Le=features_extractor_kwargs['Le'],
                P=features_extractor_kwargs['P'],
                tda=features_extractor_kwargs['tda'],
                device=device
            ).to(device=device)
        elif features_extractor_kwargs['feature_extractor'] == "MLP":
            from Feature_Extractors import MLP
            inter_dim = features_dim * (features_extractor_kwargs['K'] + 1) * features_extractor_kwargs['P']
            self.features_extractor = MLP(
                node_dim=node_dim,
                features_dim=features_dim,
                inter_dim=inter_dim,
                device=device
            ).to(device=device)
        elif features_extractor_kwargs['feature_extractor'] == "AM":
            from Feature_Extractors import GraphAttentionEncoder
            self.features_extractor = GraphAttentionEncoder(
                node_dim=node_dim,
                n_heads=features_extractor_kwargs['n_heads'],
                embed_dim=features_dim,
                n_layers=features_extractor_kwargs['Le'],
                device=device
            ).to(device=device)
        self.log_std_init = log_std_init = 0.0
        self.agent_decision_context = th.nn.Linear(agent_node_dim, features_dim).to(device=device)
        self.agent_context = th.nn.Linear(agent_node_dim, features_dim).to(device=device)
        self.full_context_nn = th.nn.Linear(3 * features_dim + 2, features_dim).to(device=device)
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

        self.project_fixed_context = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.project_node_embeddings = th.nn.Linear(features_dim, 3 * features_dim, bias=False).to(device=device)
        self.project_out = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.n_heads = features_extractor_kwargs['n_heads']
        self.tanh_clipping = features_extractor_kwargs['tanh_clipping']
        self.mask_logits = features_extractor_kwargs['mask_logits']
        self.temp = features_extractor_kwargs['temp']
        self._build()
        self.batch_norm = th.nn.BatchNorm1d(128)
        self.network = th.nn.Sequential(
            th.nn.Linear(features_dim, 64),
            th.nn.Tanh()
        )
        self.final_output = th.nn.Tanh()
        self.trainable_output = th.nn.Linear(64, 1, bias=False)
        #th.nn.init.uniform_(self.trainable_output.weight, -0.001, 0.001)

        self.bo1 = th.nn.BatchNorm1d(128)
       # self.bo2 = th.nn.BatchNorm1d(64)
        # non-trainable part
       # self.non_trainable_output_weights = th.nn.Parameter(torch.zeros(2, 64), requires_grad=False)
       # self.non_trainable_output_bias = th.nn.Parameter(torch.zeros(2), requires_grad=True)
        

        
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        actions, values, log_prob = self.forward(observation, deterministic=deterministic)
        return actions

    def _build(self):
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=128, log_std_init=self.log_std_init
            )
        

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features, graph_embed = self.extract_features(obs)
        latent_pi, values = self.context_extractor(graph_embed, obs)

        latent_pi = self.forward_actor(latent_pi)
        #print(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy() 


    def forward(self, obs, deterministic=False,  *args, **kwargs):
        features, graph_embed = self.extract_features(obs)

        latent_pi, values = self.context_extractor(graph_embed, obs)
       # print("aaaa")
       # print(latent_pi)
        latent_pi = self.forward_actor(latent_pi)
       # print(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")



    def get_distribution(self, obs):
        features, graph_embed = self.extract_features(obs)
        latent_pi, values = self.context_extractor(graph_embed, obs)
        latent_pi = self.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi)

    def context_extractor(self, graph_embed, observations):
        device = torch.device('cuda:0')
        agent_taking_decision = (observations['agent_taking_decision'].view(-1).to(torch.int64)).to(device=device)
        n_data = agent_taking_decision.shape[0]
        observations['agents_graph_nodes'] = observations['agents_graph_nodes'].to(device)
        observations['agent_talents'] = observations['agent_talents'].to(torch.float32).to(device)
        agent_taking_decision_state = (observations['agents_graph_nodes'][torch.arange(0, n_data), agent_taking_decision, :][:, None, :]).to(torch.float32)
        #print(observations['agent_talents'].dtype)
        #print(observations['agents_graph_nodes'].dtype)
        
        context = self.full_context_nn(
                    th.cat((observations['agent_talents'],graph_embed[:, None, :],self.agent_decision_context(agent_taking_decision_state),
                            self.agent_context(observations['agents_graph_nodes'].to(torch.float32)).sum(1)[:,None,:]), -1)).to(device)
        return context, self.value_net(context)

    def forward_actor(self, obs):
        obs = obs.view(-1, 128)
        obs = self.bo1(obs)
        x = self.network(obs)
        #x = self.bo2(x)
        trainable_part = self.trainable_output(x)
        #non_trainable_part = F.linear(x, self.non_trainable_output_weights, self.non_trainable_output_bias)
       # combined_output = torch.cat((non_trainable_part, trainable_part ), dim=1)
        #print(combined_output)
        sigmoid_output = self.final_output(trainable_part)
        return sigmoid_output

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features, graph_embed = self.extract_features(obs)

        latent_pi, values = self.context_extractor(graph_embed, obs)
        return values