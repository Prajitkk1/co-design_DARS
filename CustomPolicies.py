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
from stable_baselines3.common.distributions import Distribution
from torch.distributions import Bernoulli, Categorical, Normal

#   TODO:
#   Make the policy network task independent


class MHA_Decoder(th.nn.Module):

    def __init__(self,
                 features_dim,
                 features_extractor_kwargs,
                 device):
        super(MHA_Decoder, self).__init__()
        self.project_fixed_context = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.project_node_embeddings = th.nn.Linear(features_dim, 3 * features_dim, bias=False).to(device=device)
        self.project_out = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.n_heads = features_extractor_kwargs['n_heads']
        self.tanh_clipping = features_extractor_kwargs['tanh_clipping']
        self.mask_logits = features_extractor_kwargs['mask_logits']
        self.temp = features_extractor_kwargs['temp']

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = th.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # if self.mask_inner:
        #     assert self.mask_logits, "Cannot mask inner without masking logits"
        #     compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = th.matmul(th.softmax(compatibility, dim=-1), glimpse_V)
        # heads = th.matmul(th.softmax(compatibility, dim=-3), glimpse_V)
        # heads = th.matmul(compatibility, glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = (th.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))) ### steve has made a change here (normalizing)

        ## From the logits compute the probabilities by clipping, masking and softmax
        # if self.tanh_clipping > 0:
        #     logits = th.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[th.tensor(mask[:,:,:].reshape(logits.shape), dtype=th.bool)] = -math.inf
        # if mask[0, 0,0] == 1:
        #     logits[:,:,0] = -math.inf
            # print("depot masked")
        # print("Shape of logits: ", logits.shape)
        return logits

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )


    def _get_attention_node_data(self, fixed):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def forward(self, latent_pi, features, obs, num_steps=1):
        fixed_context = self.project_fixed_context(latent_pi)

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(features[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        fixed = AttentionModelFixed(features, fixed_context, *fixed_attention_node_data)
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        query = fixed.context_node_projected #+ latent_pi
        log_p = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, obs['mask'])

        log_p = th.log_softmax(log_p / self.temp, dim=-1)
        #print(log_p.shape, "logppppp")
        return log_p

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
        self.action_dist = HybridDistribution()


        self.project_fixed_context = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.project_node_embeddings = th.nn.Linear(features_dim, 3 * features_dim, bias=False).to(device=device)
        self.project_out = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.n_heads = features_extractor_kwargs['n_heads']
        self.tanh_clipping = features_extractor_kwargs['tanh_clipping']
        self.mask_logits = features_extractor_kwargs['mask_logits']
        self.temp = features_extractor_kwargs['temp']
        self.action_decoder = MHA_Decoder(features_dim=features_dim,
                                               features_extractor_kwargs=features_extractor_kwargs, device=device)
        self._build(device)
        self.batch_norm = th.nn.BatchNorm1d(128)

        self.final_output = th.nn.Sigmoid()
        # non-trainable part
        self.non_trainable_output_weights = th.nn.Parameter(torch.zeros(2, features_dim), requires_grad=False)
        self.non_trainable_output_bias = th.nn.Parameter(torch.zeros(2), requires_grad=True)
        param_groups = [
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if n != 'non_trainable_output_bias' and
                    p is not self.log_std
                ],
                'lr': lr_schedule(1)
            },
            {
                'params': [self.non_trainable_output_bias],
                'lr': 0.001  
            },
            {
                'params': [self.log_std],
                'lr': 0.1  
            }
        ]

        self.optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)
        
        #self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        actions, values, log_prob = self.forward(observation, deterministic=deterministic)
        return actions

    def _build(self,device):
        self.log_std = self.action_dist.proba_distribution_net(
        latent_dim=2, log_std_init=self.log_std_init , device= device
                )   
        

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features, graph_embed = self.extract_features(obs)
        latent_pi, values = self.context_extractor(graph_embed, obs)
        logits = self.action_decoder(latent_pi, features, obs).squeeze(1)
        non_trainable_part = F.linear(latent_pi, self.non_trainable_output_weights, self.non_trainable_output_bias)
        morphology = self.final_output(non_trainable_part).squeeze(1)
        distribution = self._get_action_dist_from_latent(morphology, logits)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy() 


    def forward(self, obs, deterministic=False,  *args, **kwargs):
        features, graph_embed = self.extract_features(obs)
        latent_pi, values = self.context_extractor(graph_embed, obs)
        logits = self.action_decoder(latent_pi, features, obs).squeeze(1)

        non_trainable_part = F.linear(latent_pi, self.non_trainable_output_weights, self.non_trainable_output_bias)
        morphology = self.final_output(non_trainable_part).squeeze(1)
        distribution = self._get_action_dist_from_latent(morphology, logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, logits):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = latent_pi
        return self.action_dist.proba_distribution(mean_actions, self.log_std, logits )




    def context_extractor(self, graph_embed, observations):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        agent_taking_decision = (observations['agent_taking_decision'].view(-1).to(torch.int64)).to(device=device)
        n_data = agent_taking_decision.shape[0]
        observations['agents_graph_nodes'] = observations['agents_graph_nodes'].to(device)
        observations['agent_talents'] = observations['agent_talents'].to(torch.float32).to(device)
        agent_taking_decision_state = (observations['agents_graph_nodes'][torch.arange(0, n_data), agent_taking_decision, :][:, None, :]).to(torch.float32)

        
        context = self.full_context_nn(
                    th.cat((observations['agent_talents'],graph_embed[:, None, :],self.agent_decision_context(agent_taking_decision_state),
                            self.agent_context(observations['agents_graph_nodes'].to(torch.float32)).sum(1)[:,None,:]), -1)).to(device)
        return context, self.value_net(context)


    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features, graph_embed = self.extract_features(obs)

        latent_pi, values = self.context_extractor(graph_embed, obs)
        return values


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor




class HybridDistribution(Distribution):
    def __init__(self):
        super(HybridDistribution, self).__init__()
        self.continuous_distribution = None
        self.discrete_distribution = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0, device: str = "cuda:0") -> Tuple[th.nn.Module, th.nn.Parameter]:
        log_std = th.nn.Parameter(th.ones(2) * log_std_init, requires_grad=True)
        return log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor, logits: th.Tensor) -> "HybridDistribution":
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.continuous_distribution = Normal(mean_actions, action_std)
        self.discrete_distribution = Categorical(logits=logits)
        return self


    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        continuous_actions, discrete_action = actions[:, :2], actions[:, 2]
        continuous_log_prob = self.continuous_distribution.log_prob(continuous_actions)
        continuous_log_prob = sum_independent_dims(continuous_log_prob)
        discrete_log_prob = self.discrete_distribution.log_prob(discrete_action)
        return continuous_log_prob + discrete_log_prob




    def sample(self) -> th.Tensor:
        continuous_actions = self.continuous_distribution.sample().squeeze(1) 
        discrete_action = self.discrete_distribution.sample().reshape(-1,1)
        return th.cat([continuous_actions, discrete_action], dim=1)

    def mode(self) -> th.Tensor:
        continuous_actions = self.continuous_distribution.mean.squeeze(1) 
        discrete_action = self.discrete_distribution.probs.argmax(dim=1).reshape(-1,1)
        return th.cat([continuous_actions, discrete_action], dim=1)

    def entropy(self) -> Optional[th.Tensor]:
        continuous_entropy = sum_independent_dims(self.continuous_distribution.entropy())
        discrete_entropy = self.discrete_distribution.entropy()
        return continuous_entropy + discrete_entropy

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor,logits, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, logits) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std, logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob