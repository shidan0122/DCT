import numpy as np
import torch
import torch.nn as nn
import dgl
import scipy.spatial
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel

def labels_affnty(labels_1):
    if (isinstance(labels_1, torch.LongTensor) or
        isinstance(labels_1, torch.Tensor)):
        labels_1 = labels_1.cpu().numpy()

    if labels_1.ndim == 1:
        affnty = np.float32(labels_1 == labels_1[:, np.newaxis])
    else:
        affnty = np.float32(np.sign(np.dot(labels_1, labels_1.T)))
    return torch.Tensor(affnty)

def euclidean_dist(x, y):
    """
    Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
    Returns:
    dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    return dist

def get_graph(feature, feature1):
    similarity = (100.0 * feature @ feature.T).softmax(dim=-1)

    '''topk selection'''
    batch_size = feature.shape[0]
    topk = batch_size-10
    similarity = similarity.cpu()
    similarity = similarity.detach().numpy()
    topk_min = np.argsort(similarity, axis=-1)[:,:topk]
    for col_idx in topk_min.T:
        similarity[np.arange(similarity.shape[0]), col_idx]= 0.0

    '''label guide'''
    similarity = torch.from_numpy(similarity).cuda()
    similarity_non = np.nonzero(similarity.cuda()) #nearest neighbors index, [batchsize*number of neighbors,2]
    u = similarity_non[:,0]
    v = similarity_non[:,1]
    g = dgl.graph((u,v))

    g.ndata['feat'] = feature1
    g = g.remove_self_loop().add_self_loop()
    g.create_formats_()
    return g

def to_one_hot(x, classes):
    if len(x.shape) == 1 or x.shape[1] == 1:
        one_hot = (classes.reshape([1, -1]) == x.reshape([-1, 1])).astype('float32')
        labels = one_hot
        y = torch.tensor(labels)
    else:
        y = torch.tensor(x.astype('float32'))
    return y

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

def fx_calc_map_label(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort() # [batch, batch]
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)


def zero2eps(x):
    x[x == 0] = 1
    return x
def normalize(affnty):
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affnty, axis=0))

    out_affnty = affnty/col_sum
    in_affnty = np.transpose(affnty/row_sum)
    return in_affnty, out_affnty
# construct affinity matrix via rbf kernel
def rbf_affnty(X, Y, topk=10):
    X = X.cpu()
    Y = Y.cpu()
    X = X.detach().numpy()
    Y = Y.detach().numpy()

    rbf_k = rbf_kernel(X, Y)
    print (rbf_k)
    topk_max = np.argsort(rbf_k, axis=1)[:,-topk:]

    affnty = np.zeros(rbf_k.shape)
    for col_idx in topk_max.T:
        affnty[np.arange(rbf_k.shape[0]), col_idx] = 1.0
    similarity_non = np.nonzero(affnty)
    u = similarity_non[:, 0]
    v = similarity_non[:, 1]
    g = dgl.graph((u, v))
    g.ndata['feat'] = X
    g = g.remove_self_loop().add_self_loop()
    g.create_formats_()
    return g 
