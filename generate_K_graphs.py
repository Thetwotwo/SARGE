from dataset import load

import numpy as np
import torch

import warnings

warnings.filterwarnings('ignore')

dataset = "citeseer"
num = 3000

graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(dataset)

N = graph.number_of_nodes()

I = torch.eye(N, dtype=torch.float)
adj = graph.adjacency_matrix().to_dense() + I
d = torch.sum(adj, dim=1)
d_inv = 1 / d
d_inv[d_inv == np.inf] = 0.0
D_inv = torch.diag(d_inv)
D = torch.diag(d)
print("max degree", d.max())

A = adj.float()
A_temp = I
torch.save(A_temp, "./graphs/" + dataset  + "-graphs/"+"1.pt")
for i in range(2, num):
    A_current = torch.mm(A, A_temp).bool().float()
    torch.save(A_current-A_temp, "./graphs/" + dataset  + "-graphs/"+str(i) + ".pt")
    A_temp = A_current