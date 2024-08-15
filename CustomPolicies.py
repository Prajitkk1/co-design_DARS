"""
Author: Steve Paul 
Date: 1/18/22 """
from argparse import Action
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
import random
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
        value_net_net1 = [th.nn.Linear(2, 6, bias=True),
                         th.nn.Linear(6, 1, bias=True)]
        self.value_net1 = th.nn.Sequential(*value_net_net1).to(device=device)
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
        self.non_trainable_output_bias = th.nn.Parameter(torch.randn(2), requires_grad=True)
        morphology_network = [th.nn.Linear(2, 12, bias=True),
                         th.nn.Linear(12, 2, bias=True)]
        self.morphology_network = th.nn.Sequential(*morphology_network).to(device=device)

        #self.optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        #self.optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)
        

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
        morphology = self.morphology_network(non_trainable_part)
        morphology = self.final_output(morphology).squeeze(1)
        distribution = self._get_action_dist_from_latent(morphology, logits)

        log_prob = distribution.log_prob(actions)
        #print(log_prob, "log_prob")
        return values, log_prob, distribution.entropy() 


    def forward(self, obs, deterministic=False, *args, **kwargs):
        features, graph_embed = self.extract_features(obs)
        latent_pi, values = self.context_extractor(graph_embed, obs)
        logits = self.action_decoder(latent_pi, features, obs).squeeze(1)

        non_trainable_part = F.linear(latent_pi, self.non_trainable_output_weights, self.non_trainable_output_bias)
        morphology = self.morphology_network(non_trainable_part)
        #print(morphology)
        morphology = self.final_output(morphology).squeeze(1)
        #print(morphology)
        distribution = self._get_action_dist_from_latent(morphology, logits)
        generated_actions = distribution.get_actions(obs = obs, deterministic=deterministic)
        #print(generated_actions)
        actions_to_clip = generated_actions[:, :2]  # Selects the actions at indices 0 and 1 along dimension 1

        # Clip the selected actions
        clipped_actions = actions_to_clip.clamp(0.0, 1.0)

        # Replace the original actions with the clipped actions
        generated_actions[:, :2] = clipped_actions
        talent_actions = obs["talent_beginned"].to(device = generated_actions.device)
        #print(talent_actions, "talent_actions")
        #print(obs["step"], "obsstep")
        # Checking whether steps == 1 for each item in the batch
        condition = obs["step"] >= 1  # Shape: [batch_size, 1]

        # Update only the first two elements of each action in generated_actions based on condition
        mask = condition.expand(-1, 2).to(device = generated_actions.device)  # Shape: [batch_size, 2], broadcasted to match action dims
        zeros = torch.zeros_like(generated_actions[:, 2:3]).to(device = generated_actions.device)  # Shape: [batch_size, 1], to keep the third element unchanged

        # Construct a mask with True values for elements to be replaced
        replacement_mask = torch.cat((mask, zeros.bool()), dim=1).to(device = generated_actions.device)  # Shape: [batch_size, 3]

        # Extend talent_actions to match the action dimensions
        replacements = torch.cat((talent_actions, zeros), dim=1)  # Shape: [batch_size, 3]

        # Replace the first two elements of each action in generated_actions where condition is True
        actions = torch.where(replacement_mask.unsqueeze(-1), replacements.unsqueeze(-1), generated_actions.unsqueeze(-1)).squeeze(-1)
        #print(actions, "actions in forward")
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
  
        value1 = self.value_net1(observations['agent_talents'])
        return context, self.value_net(context)+value1


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
        #print(action_std)
        self.continuous_distribution = Normal(mean_actions, action_std)
        self.discrete_distribution = Categorical(logits=logits)
        return self


    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        continuous_actions, discrete_action = actions[:, :2], actions[:, 2]
        #print(continuous_actions, "actions")
        #print(discrete_action, "discrete_action")
        continuous_log_prob = self.continuous_distribution.log_prob(continuous_actions)
        #print(continuous_log_prob, "log_prob")
        continuous_log_prob = sum_independent_dims(continuous_log_prob)
        #print(continuous_log_prob, "log_prob_summation")
        discrete_log_prob = self.discrete_distribution.log_prob(discrete_action)
        return continuous_log_prob + discrete_log_prob

    def sample(self, epsilon: float = 0.5) -> th.Tensor:
        continuous_actions = self.continuous_distribution.sample().squeeze(1) 

        discrete_action = self.discrete_distribution.sample().reshape(-1, 1)
        #discrete_action = self.discrete_distribution.sample().reshape(-1, 1)

        return th.cat([continuous_actions, discrete_action], dim=1)
    
    def sample1(self, obs, epsilon: float = 0.25) -> th.Tensor:
        continuous_actions = self.continuous_distribution.sample().squeeze(1)
        mask = th.tensor(obs["mask"], dtype=th.bool, device=continuous_actions.device).squeeze(-1)  
    
        batch_size, num_actions = mask.shape
        discrete_actions = th.empty((batch_size, 1), dtype=th.long, device=continuous_actions.device)

        if th.rand(1) < epsilon:
            for i in range(batch_size):
                valid_actions = th.nonzero(mask[i] == 0).view(-1)
                print(valid_actions, "valid actions")
                print(mask[i], "mask")
                if len(valid_actions) > 0:
                    # Select a random index from the valid actions for this batch
                    random_action = valid_actions[th.randint(0, len(valid_actions), (1,))]
                    discrete_actions[i] = random_action
                else:
                    # Handle the case where there are no valid actions
                    raise ValueError(f"No valid action found in obs['mask'] for batch {i}.")
                print("action", random_action)
            #discrete_action1 = self.discrete_distribution.sample().reshape(-1, 1)
            #print(discrete_actions.shape, discrete_action1.shape)
        else:
            # Sample from the discrete distribution for each batch
            discrete_actions = self.discrete_distribution.sample().reshape(-1, 1)

        return th.cat([continuous_actions, discrete_actions], dim=1)

    def mode(self) -> th.Tensor:
        continuous_actions = self.continuous_distribution.mean.squeeze(1) 
        discrete_action = self.discrete_distribution.probs.argmax(dim=1).reshape(-1,1)
        return th.cat([continuous_actions, discrete_action], dim=1)

    def entropy(self) -> Optional[th.Tensor]:
        cont_ent = self.continuous_distribution.entropy()
        continuous_entropy = sum_independent_dims(cont_ent)
        discrete_entropy = self.discrete_distribution.entropy()
        #print(continuous_entropy, discrete_entropy)
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