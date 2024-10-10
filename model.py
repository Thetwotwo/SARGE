import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class LogReg(nn.Module):
    def __init__(self, embed_dim, num_c):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(embed_dim, num_c)

    def forward(self, x):
        ret = self.linear(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class MLPsingle(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLPsingle, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x



class Pipeline(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp = False, coeff = None):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.decoder = nn.Linear(out_dim, in_dim)
        self.coeff = nn.Parameter(torch.ones((1, 1, 3)) * torch.tensor([1, coeff[0], coeff[1]]) ) 
        
    def get_embedding(self, graph, feat):
        out = self.backbone(graph, feat)
        return out.detach()

    def forward(self, graph, feat):
        h = self.backbone(graph, feat)
        z = (h - h.mean(0)) / h.std(0)    
        return h, z

    
    def reconstruct(self, Laps, repre):       
        f_fre = torch.sum(self.coeff * Laps, dim=2)
        x_hat = f_fre @ self.decoder(repre) 
        return x_hat





