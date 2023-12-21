"""
Author: Steve Paul 
Date: 1/18/22 """

from torch import nn
import torch
import math
import numpy as np
from typing import Union

class MLP(nn.Module):

    def __init__(self,
                 n_layers=2,
                 features_dim=128,
                 node_dim=2,
                 inter_dim=128,
                 device: Union[torch.device, str] = "auto"
                 ):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.features_dim = features_dim
        self.node_dim = node_dim
        print([node_dim, inter_dim])
        self.init_embed = nn.Linear(node_dim, inter_dim)
        self.layer_1 = nn.Linear(inter_dim, inter_dim)
        self.WF = nn.Linear(inter_dim, features_dim)
        self.init_embed_depot = nn.Linear(2, features_dim)
        self.activ = nn.Tanh()

    def forward(self, data, mask=None):
        X = data['task_graph_nodes'].to(device=self.init_embed.weight.device)
        
        F0 = self.init_embed(X)
        F0 = self.activ(self.layer_1(F0))
        F0 = self.activ(self.WF(F0))
        init_depot_embed = self.init_embed_depot(data['depot'].to(device=self.init_embed.weight.device))[:]
        h = torch.cat((init_depot_embed, F0), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class CAPAM(nn.Module):

    def __init__(self,
                 Le=2,
                 features_dim=128,
                 P=4,
                 node_dim=2,
                 K=3,
                 tda=False,
                 device: Union[torch.device, str] = "cpu"
                 ):
        super(CAPAM, self).__init__()
        self.Le = Le
        self.features_dim = features_dim
        self.P = P
        self.K = K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, features_dim * P)
        self.init_embed_depot = nn.Linear(2, features_dim)
        self.device = device

        self.W_L_1_G1 = nn.Linear(features_dim * (K + 1) * P, features_dim)
        self.W_L_1_G2 = nn.Linear(features_dim * (K + 1) * P, features_dim)
        self.W_L_1_G3 = nn.Linear(features_dim * (K + 1) * P, features_dim)
        self.W_L_1_G4 = nn.Linear(features_dim * (K + 1) * P, features_dim)

        self.W_F = nn.Linear(features_dim * P, features_dim)
        self.tda = tda

        self.activ = nn.Tanh()

    def data_loader(self, data, batch_size, index):
        X = data['task_graph_nodes'][index:index+batch_size, :,:]
        num_samples, num_locations, _ = X.size()
        #print([num_samples, num_locations])
        if self.tda:
            L = data["topo_laplacian"][index:index+batch_size, :,:]
        else:
            A = data['task_graph_adjacency'][index:index+batch_size, :,:]
            # A = data['task_graph_adjacency']
            D = torch.mul(torch.eye(num_locations, device=X.device).expand((num_samples, num_locations, num_locations)),
                          (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
            L = D - A

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
        F0_cube= torch.mul(F0_squared[:, :, :], F0[:, :, :])
        # K = 3

        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L_squared, L)
        # L_cube = torch.matmul(L, L_squared)  torch.cat([torch.matmul(L**(i), F0)[:, :, :] for i in range(self.K+1)], dim=-1)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :],
                                          torch.matmul(L_cube, F0)[:, :, :],
                                          ),
                                         -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                          torch.matmul(L, F0_squared)[:, :, :],
                                          torch.matmul(L_squared, F0_squared)[:, :, :],
                                          torch.matmul(L_cube, F0_squared)[:, :, :],
                                          ),
                                         -1))
        g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :],
                                          torch.matmul(L, F0_cube)[:, :, :],
                                          torch.matmul(L_squared, F0_cube)[:, :, :],
                                          torch.matmul(L_cube, F0_cube)[:, :, :],
                                          ),
                                         -1))

        g_L1_4 = self.W_L_1_G4(torch.cat((F0_cube[:, :, :],
                                          torch.matmul(L, F0_cube)[:, :, :],
                                          torch.matmul(L_squared, F0_cube)[:, :, :],
                                          torch.matmul(L_cube, F0_cube)[:, :, :],
                                          ),
                                         -1))

        # g_L1_1 = self.W_L_1_G1(torch.cat([torch.matmul(L**i, F0)[:, :, :] for i in range(self.K+1)], dim=-1))

        # F1 = self.normalization_1(F1)
        F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3, g_L1_4), -1)
        F1 = self.activ(F1)  # + F0
        # F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))
        init_depot_embed = self.init_embed_depot(data['depot'][index:index+batch_size, :,:])[:]
        h = torch.cat((init_depot_embed, F_final), 1)
        return h, index + batch_size

    def forward(self, data, mask=None):
        X = data['task_graph_nodes'].to(device=self.init_embed.weight.device)
        num_samples, num_locations, _ = X.size()
        #print([num_samples, num_locations])
        if num_samples == 1:
            if self.tda:
                L = data["topo_laplacian"]
            else:
                A = data['task_graph_adjacency'].to(device=self.init_embed.weight.device)
                D = torch.mul(
                    torch.eye(num_locations, device=X.device).expand((num_samples, num_locations, num_locations)),
                    (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
                L = D - A

            F0 = self.init_embed(X)
            F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
            F0_cube = torch.mul(F0_squared[:, :, :], F0[:, :, :])
            # K = 3

            L_squared = torch.matmul(L, L)
            L_cube = torch.matmul(L_squared, L)
            # L_cube = torch.matmul(L, L_squared)  torch.cat([torch.matmul(L**(i), F0)[:, :, :] for i in range(self.K+1)], dim=-1)

            g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                              torch.matmul(L, F0)[:, :, :],
                                              torch.matmul(L_squared, F0)[:, :, :],
                                              torch.matmul(L_cube, F0)[:, :, :],
                                              ),
                                             -1))
            g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                              torch.matmul(L, F0_squared)[:, :, :],
                                              torch.matmul(L_squared, F0_squared)[:, :, :],
                                              torch.matmul(L_cube, F0_squared)[:, :, :],
                                              ),
                                             -1))
            g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :],
                                              torch.matmul(L, F0_cube)[:, :, :],
                                              torch.matmul(L_squared, F0_cube)[:, :, :],
                                              torch.matmul(L_cube, F0_cube)[:, :, :],
                                              ),
                                             -1))
            g_L1_4 = self.W_L_1_G4(torch.cat((F0_cube[:, :, :],
                                              torch.matmul(L, F0_cube)[:, :, :],
                                              torch.matmul(L_squared, F0_cube)[:, :, :],
                                              torch.matmul(L_cube, F0_cube)[:, :, :],
                                              ),
                                             -1))

            # g_L1_1 = self.W_L_1_G1(torch.cat([torch.matmul(L**i, F0)[:, :, :] for i in range(self.K+1)], dim=-1))

            # F1 = self.normalization_1(F1)
            F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3, g_L1_4), -1)
            F1 = self.activ(F1)  # + F0
            F_final = self.activ(self.W_F(F1))
            init_depot_embed = self.init_embed_depot(data['depot'].to(device=self.init_embed.weight.device))[:]
            h = torch.cat((init_depot_embed, F_final), 1)
        else:

            index = 0
            batch_size = 1000
            h, index = self.data_loader(data, batch_size, index)
            while index < num_samples:
                h_batch, index = self.data_loader(data, batch_size, index)
                h = torch.cat((h, h_batch), dim=0)
        # print("Shape of the node embeddings: ", h.shape)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class CAPAM_P(nn.Module):
    def __init__(self,
                 Le=2,
                 features_dim=128,
                 P=1,
                 node_dim=2,
                 K=1,
                 device: Union[torch.device, str] = "auto"
                 ):
        super(CAPAM_P, self).__init__()
        self.Le = Le
        self.features_dim = features_dim
        self.P = P
        self.K = K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, features_dim).to(device=device)
        self.init_embed_depot = nn.Linear(2, features_dim).to(device=device)
        self.device = device
        graph_capsule_layers = [GraphCapsule(P=P, K=K, features_dim=features_dim, device=device) for _ in range(Le)]
        self.graph_capsule_layers = nn.Sequential(*graph_capsule_layers).to(device=device)
        self.activ = nn.Tanh()


    def forward(self, data):
        X = data["task_graph_nodes"]
        num_samples, num_locations, _ = X.size()
        if num_samples == 1:
            A = data['task_graph_adjacency']
            D = torch.mul(torch.eye(num_locations, device=X.device).expand((num_samples, num_locations, num_locations)),
                          (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
            F0 = self.init_embed(X)
            L = D - A
            init_depot_embed = self.init_embed_depot(data['depot'])
            F = self.graph_capsule_layers({"embeddings": F0, "L": L})["embeddings"]
            h = torch.cat((init_depot_embed, F), 1)
        else:
            index = 0
            batch_size = 2000
            h, index = self.data_loader(data, batch_size, index)
            while index < num_samples:
                h_batch, index = self.data_loader(data, batch_size, index)
                h = torch.cat((h,h_batch), dim=0)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

    def data_loader(self, data, batch_size, index):
        X = data['task_graph_nodes'][index:index+batch_size, :,:]
        num_samples, num_locations, _ = X.size()
        A = data['task_graph_adjacency'][index:index+batch_size, :,:]
        D = torch.mul(torch.eye(num_locations, device=X.device).expand((num_samples, num_locations, num_locations)),
                      (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        F0 = self.init_embed(X)
        L = D - A
        init_depot_embed = self.init_embed_depot(data['depot'][index:index+batch_size, :,:])
        F = self.graph_capsule_layers({"embeddings": F0, "L": L})["embeddings"]
        h = torch.cat((init_depot_embed, F), 1)
        return h, index + batch_size


class GraphCapsule(nn.Module):
    def __init__(self,
                 features_dim=128,
                 P=1,
                 K=1,
                 device: Union[torch.device, str] = "auto"
                 ):
        super(GraphCapsule, self).__init__()
        self.features_dim = features_dim
        self.P = P
        if P == 1:
            self.conv1 = Conv(P=P, K=K, features_dim=features_dim, device=device)
        elif P == 2:
            self.conv1 = Conv(P=P, K=K, features_dim=features_dim, device=device)
            self.conv2 = Conv(P=P, K=K, features_dim=features_dim, device=device)
        elif P == 3:
            self.conv1 = Conv(P=P, K=K, features_dim=features_dim, device=device)
            self.conv2 = Conv(P=P, K=K, features_dim=features_dim, device=device)
            self.conv3 = Conv(P=P, K=K, features_dim=features_dim, device=device)
        elif P == 4:
            self.conv1 = Conv(P=P, K=K, features_dim=features_dim, device=device)
            self.conv2 = Conv(P=P, K=K, features_dim=features_dim, device=device)
            self.conv3 = Conv(P=P, K=K, features_dim=features_dim, device=device)
            self.conv4 = Conv(P=P, K=K, features_dim=features_dim, device=device)

        # self.conv = [Conv(P=P, K=K, features_dim=features_dim, device=device) for _ in range(P)]
        self.W_F = nn.Linear(features_dim * P*(K+1), features_dim).to(device=device)
        self.activ = nn.Tanh()

    def forward(self, data):
        X = data["embeddings"]
        L = data['L'].to(device=X.device)
        # torch.cat([self.conv[p - 1]({"embeddings": X ** p, "L": L}) for p in range(1, self.P + 1)],
        #           dim=-1)
        if self.P == 1:
            capsule_func =  self.conv1({"embeddings": X ** 1, "L": L})
        elif self.P == 2:
            capsule_func = torch.cat((self.conv1({"embeddings": X ** 1, "L": L}),
                                      self.conv2({"embeddings": X ** 2, "L": L})), dim=-1)
        elif self.P == 3:
            capsule_func = torch.cat((self.conv1({"embeddings": X ** 1, "L": L}),
                                      self.conv2({"embeddings": X ** 2, "L": L}),
                                      self.conv3({"embeddings": X ** 3, "L": L})), dim=-1)
        elif self.P == 4:
            capsule_func = torch.cat((self.conv1({"embeddings": X ** 1, "L": L}),
                                      self.conv2({"embeddings": X ** 2, "L": L}),
                                      self.conv3({"embeddings": X ** 3, "L": L}),
                                      self.conv4({"embeddings": X ** 3, "L": L})), dim=-1)
        return {"L": L,
                "embeddings":
                    self.activ(self.W_F(capsule_func))
                }


class Conv(nn.Module):
    def __init__(self,
                 P=1,
                 features_dim=128,
                 K=1,
                 device: Union[torch.device, str] = "auto"
                 ):
        super(Conv, self).__init__()
        self.features_dim = features_dim
        self.K = K
        # self.W = nn.Linear(features_dim * (K + 1)*P, features_dim).to(device=device)
        self.activ = nn.Tanh()

    def forward(self, data):
        X = data["embeddings"]
        L = data["L"].to(device=X.device)
        return torch.cat([torch.matmul(L**i, X) for i in range(self.K+1)], dim=-1)


class GCAPCNFeatureExtractor(nn.Module):

    def __init__(self,
                 n_layers=2,
                 features_dim=128,
                 n_p=1,
                 node_dim=2,
                 n_K=1
                 ):
        super(GCAPCNFeatureExtractor, self).__init__()
        self.n_layers = n_layers
        self.n_dim = features_dim
        self.features_dim=features_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, features_dim * n_p)
        self.init_embed_depot = nn.Linear(2, features_dim)

        self.W_L_1_G1 = nn.Linear(features_dim * (n_K + 1) * n_p, features_dim)

        self.normalization_1 = nn.BatchNorm1d(features_dim * n_p)

        self.W_F = nn.Linear(features_dim * n_p, features_dim)
        self.agent_decision_context = torch.nn.Linear(2, features_dim)
        self.agent_context = torch.nn.Linear(2, features_dim)
        self.agent_mask_encoding = torch.nn.Linear(11, features_dim)

        self.activ = nn.Tanh()

    def forward(self, data, mask=None):
        # active_tasks = ((data['nodes_visited'] == 0).nonzero())[:,1]

        X = data['task_graph_nodes']
        # X = X[:,active_tasks[1:]-1,:]
        # distance_matrix = ((((X[:, :, None] - X[:, None]) ** 2).sum(-1)) ** .5)[0]



        num_samples, num_locations, _ = X.size()

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)

        # K = 3
        # L = D - A
        L_topo = data["topo_laplacian"]
        L = L_topo
        # L_squared = torch.matmul(L, L)
        # L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]
                                          # torch.matmul(L_squared, F0)[:, :, :]
                                          ),
                                         -1))


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) #+ F0
        # F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:]
        h = torch.cat((init_depot_embed, F_final), 1)
        # print("Shape of the node embeddings: ", h.shape)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
            device: Union[torch.device, str] = "auto"
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.init_embed_depot = nn.Linear(2, embed_dim)

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.initial_embed(x)

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

    def initial_embed(self, data):

        return torch.cat(
            (
                self.init_embed_depot(data['depot'])[:, :],
                self.init_embed(data['task_graph_nodes'])
            ),
            1
        )


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input
