"""
Implementation of the sc with VAE Models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvae.modules.model import Encoder, Decoder
from scvae.models.GATlayers import GraphAttentionLayer, SpGraphAttentionLayer

def order_F_to_C(n):
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    idx = list(idx)
    return idx

def init_dct(n, m):
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    """See paper: Designing of overcomplete dictionaries based on DCT and DWT """
    oc_dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        oc_dictionary[:, k] = V / np.linalg.norm(V)
    oc_dictionary = np.kron(oc_dictionary, oc_dictionary)
    oc_dictionary = oc_dictionary.dot(np.diag(1 / np.sqrt(np.sum(oc_dictionary ** 2, axis=0))))
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    oc_dictionary = oc_dictionary[idx, :]
    oc_dictionary = torch.from_numpy(oc_dictionary).float()
    return oc_dictionary

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.softmax(x, dim=1).view(x.size(0), int(x.size(1)**0.5), int(x.size(1)**0.5), x.size(2))

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)
        # self.bn = nn.BatchNorm1d(nfeat)

    def forward(self, x):
        alpha = self.W(x)
        return F.softmax(alpha, dim=1).view(alpha.size(0), int(alpha.size(1)**0.5), int(alpha.size(1)**0.5), alpha.size(2))


class Model_VAEf4(nn.Module):
    def __init__(
            self,
            input_dim,
            Hidden_size,
            H_1,
            H_2,
            num_soft_thresh,
            Dict_init,
            c_init,
            w_init,
            beta,
            device,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, Hidden_size, 4, 2, 1),
            nn.BatchNorm2d(Hidden_size),
            nn.ReLU(True),
            nn.Conv2d(Hidden_size, Hidden_size, 4, 2, 1),
            ResBlock(Hidden_size),
            ResBlock(Hidden_size),
        )

        self.num_soft_thresh = num_soft_thresh
        #self.min_v = min_v
        #self.max_v = max_v

        h_s, num_atoms = Dict_init.shape
        soft_comp = torch.zeros(num_atoms).to(device)
        Identity = torch.eye(num_atoms).to(device)

        self.soft_comp = soft_comp
        self.Identity = Identity
        self.beta = beta
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)
        #self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        #Feedforward neural network to learn beta
        #self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        #self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        #self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        #self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        #EquivarianceLayer to learn alpha
        self.el1 = nn.Sequential(
            torch.nn.Linear(Hidden_size, H_1, bias=True),
            torch.nn.Sigmoid()
        )
        self.el2 = nn.Sequential(
            nn.Linear(H_1, H_2, bias=True),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

        self.w = torch.nn.Parameter(w_init)

        self.decoder = nn.Sequential(
            ResBlock(Hidden_size),
            ResBlock(Hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(Hidden_size, Hidden_size, 4, 2, 1),
            nn.BatchNorm2d(Hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(Hidden_size, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def soft_thresh(self, x, L):
        return torch.sign(x) * torch.max(torch.abs(x) - L, self.soft_comp)

    def equivarianceLayer(self, A, layer):
        # A: Input [bs, Hf, Wf, d], bs is the batch size, Hf and Hw are the height and width of feature map vectors
        # d is the dimension of the feature map vectors
        bs, Hf, Wf, d = A.size()
        A = A.flatten(start_dim=1, end_dim=2)
        x_max = torch.max(A, dim=1).values.view(bs, 1, d)

        A_minus = torch.subtract(A, x_max)  # [Nb, d]
        # [Nb, k]
        if layer == 1:
            Xw = self.el1(A_minus)
            Xw = Xw.view(bs, Hf, Hf, Xw.size()[-1])
        elif layer == 2:
            Xw = self.el2(A_minus)
            Xw = Xw.view(bs, Hf, Hf, Xw.size()[-1])
        return Xw

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()

        L = self.beta / self.c
        y = torch.matmul(z_e_x_, self.Dict)

        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, L)
        for t in range(self.num_soft_thresh):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, L)

        x_pred = torch.matmul(z, self.Dict.t())
        #x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)

        z_sdl_x = x_pred.permute(0, 3, 1, 2).contiguous()
        x_tilde = self.decoder(z_sdl_x)

        #learn alpha
        A = torch.square(z_e_x_ - x_pred)
        el1 = self.equivarianceLayer(A, 1)
        alpha = self.equivarianceLayer(el1, 2)

        rec_latent_representation = alpha * A
        return x_tilde, rec_latent_representation, self.Dict, z

class Model_VAEf16(nn.Module):
    def __init__(
            self,
            ddconfig,
            input_dim,
            Hidden_size,
            H_1,
            H_2,
            num_soft_thresh,
            Dict_init,
            c_init,
            w_init,
            beta,
            device,
            attention,
    ):
        super().__init__()

        self.attention = attention

        self.encoder = Encoder(**dict(ddconfig))

        self.num_soft_thresh = num_soft_thresh

        h_s, num_atoms = Dict_init.shape
        soft_comp = torch.zeros(num_atoms).to(device)
        Identity = torch.eye(num_atoms).to(device)
        #soft_comp = torch.zeros(num_atoms)
        #Identity = torch.eye(num_atoms)

        self.soft_comp = soft_comp
        self.Identity = Identity
        self.beta = beta
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)
        #self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        #Feedforward neural network to learn beta
        #self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        #self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        #self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        #self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        #EquivarianceLayer to learn alpha
        self.el1 = nn.Sequential(
            torch.nn.Linear(Hidden_size, H_1, bias=True),
            torch.nn.Sigmoid()
        )
        self.el2 = nn.Sequential(
            nn.Linear(H_1, H_2, bias=True),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

        self.GAT = GAT(512, 64, 1, 0.6, 0.2, 6).to(self.device)

        self.SGC = SGC(512, 1).to(self.device)

        self.w = torch.nn.Parameter(w_init)

        self.decoder = Decoder(**dict(ddconfig))

        self.apply(weights_init)

    def soft_thresh(self, x, L):
        return torch.sign(x) * torch.max(torch.abs(x) - L, self.soft_comp)

    def equivarianceLayer(self, A, layer):
        # A: Input [bs, Hf, Wf, d], bs is the batch size, Hf and Hw are the height and width of feature map vectors
        # d is the dimension of the feature map vectors
        bs, Hf, Wf, d = A.size()
        A = A.flatten(start_dim=1, end_dim=2)
        x_max = torch.max(A, dim=1).values.view(bs, 1, d)

        A_minus = torch.subtract(A, x_max)  # [Nb, d]
        if layer == 1:
            Xw = self.el1(A_minus)
            Xw = Xw.view(bs, Hf, Wf, Xw.size()[-1])
        elif layer == 2:
            Xw = self.el2(A_minus)
            Xw = Xw.view(bs, Hf, Wf, Xw.size()[-1])
        return Xw

    def adjacency_matrix(self, z):

        bs, Hf, Wf, d = z.size()
        #z = z.flatten(start_dim=1, end_dim=2)

        # Normalize the input tensor along dimension 2
        #z_normalized = z / (torch.norm(z, dim=2, keepdim=True))
        #z_normalized = torch.where(torch.isnan(z_normalized), torch.tensor(0.0), z_normalized)
        #torch.norm(z, dim=2)[0, :]

        # Calculate similarity map per the first dimension
        #z_normalized_transposed = z_normalized.permute(0, 2, 1)  # Transpose dimensions to have shape (H, W, N)
        #cosine_similarity_matrix = torch.matmul(z_normalized, z_normalized_transposed)  # Perform matrix multiplication


        # Reshape the image to be a 1D array, each element represents a node in the graph
        #nodes = z.view(-1)

        # Create a grid of coordinates for each pixel in the image
        x, y = torch.meshgrid(torch.arange(z.shape[1]), torch.arange(z.shape[2]))
        coords = torch.stack((x.reshape(-1), y.reshape(-1)), axis=-1).float()

        # Calculate the Euclidean distance between all pairs of nodes,
        # this will act as our edge weights (we will consider two nodes to be adjacent if their distance is 1)
        from scipy.spatial import distance
        distances = distance.cdist(coords, coords, 'euclidean')

        # Make adjacency matrix: nodes are adjacent if their distance is 1
        adjacency_matrix = (distances == 1)

        # Convert adjacency matrix back to torch.Tensor
        adjacency_matrix = torch.from_numpy(adjacency_matrix.astype(np.float32) + np.eye(adjacency_matrix.shape[0]).astype(np.float32))

        # Check the adjacency matrix
        # print(adjacency_matrix)

        return adjacency_matrix.to(self.device).unsqueeze(0).expand(bs, -1, -1)

    def sgc_precompute(self, features, adj, degree, alpha):
        #ori_features = features
        emb = alpha * features
        for i in range(degree):
            features = torch.matmul(adj, features)
            emb = emb + (1 - alpha) * features / degree
        return emb

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()

        L = self.beta / self.c
        y = torch.matmul(z_e_x_, self.Dict)

        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, L)
        for t in range(self.num_soft_thresh):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, L)

        x_pred = torch.matmul(z, self.Dict.t())

        z_sdl_x = x_pred.permute(0, 3, 1, 2).contiguous()
        x_tilde = self.decoder(z_sdl_x)

        #learn alpha
        if self.attention == 'eq':
            A = torch.square(z_e_x_ - x_pred)
            el1 = self.equivarianceLayer(A, 1)
            alpha = self.equivarianceLayer(el1, 2)
        elif self.attention == 'constant':
            A = torch.square(z_e_x_ - x_pred)
            bs, h, w, d = A.size()
            alpha = (torch.ones((h, w, d))/(h*w*d)).repeat(bs,1,1,1).to(self.device)
        elif self.attention == 'GAT':
            A = torch.square(z_e_x_ - x_pred)
            adj = self.adjacency_matrix(z)
            alpha = self.GAT(z.flatten(start_dim=1, end_dim=2), adj) #shape should be bs x h x w x 1
        elif self.attention == 'SSGC':
            A = torch.square(z_e_x_ - x_pred)
            adj = self.adjacency_matrix(z)
            features = self.sgc_precompute(z.flatten(start_dim=1, end_dim=2), adj, 1, 0.5)
            alpha = self.SGC(features)  # shape should be bs x h x w x 1

        rec_latent_representation = [alpha, A]
        return x_tilde, rec_latent_representation, self.Dict, z

    def get_last_layer(self):
        return self.decoder.conv_out.weight

class ToTensor(object):
    """ Convert ndarrays to Tensors. """

    def __call__(self, image):
        return torch.from_numpy(image)


class Normalize(object):
    """ Normalize the images. """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std
