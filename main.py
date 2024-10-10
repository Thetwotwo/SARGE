import argparse
import sys
from tqdm import tqdm
from model import Pipeline, LogReg
from dataset import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import warnings
import time
import os.path as osp

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='SARGE')
parser.add_argument('--name_dataset', type=str, default='cora', help='Dataset for train and test.')
parser.add_argument('--gpu_index', type=int, default=0, help='Choosen GPU index.')
parser.add_argument('--num_epochs', type=int, default=100, help='Epochs for training model.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of GCN backbone.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of logistic regression model.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of GCN backbone.')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of logistic regression model.')
parser.add_argument('--lambd', type=float, default=1, help='Coefficient for feature reconstruction.')
parser.add_argument('--beta', type=float, default=1, help='Coefficient for structure reconstruction.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN of MLP layers')
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP as backbone')
parser.add_argument("--hid_dim", type=int, default=1024, help='Hidden dimension.')
parser.add_argument("--out_dim", type=int, default=1024, help='Output dimemsion.')
parser.add_argument("--t", type=float, default=0.1, help='Diffusion time.')
parser.add_argument('--coeff', type=float, nargs='+', default=[2.0, 2.0], help='Initial coefficients of high-pass filter')
args = parser.parse_args()


if __name__ == '__main__':
    # choose gpu or cpu
    if args.gpu_index >= 0 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu_index)
    else:
        args.device = 'cpu'
    print("super-parameters are as follows: ", args)
    graph, feature, labels, num_classes, train_idx, val_idx, test_idx = load_dataset(args.name_dataset)
    in_dim = feature.shape[1]

    model = Pipeline(in_dim, args.hid_dim, args.out_dim, args.num_layers, args.use_mlp, args.coeff)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    N = graph.number_of_nodes()
    I = torch.eye(N, dtype=torch.float)
    adj = graph.adjacency_matrix().to_dense() + I
    d = torch.sum(adj, dim=1)
    d_inv = 1 / d
    d_inv[d_inv == np.inf] = 0.0
    D_inv = torch.diag(d_inv)
    D = torch.diag(d)
    D_inv_1_2 = torch.diag(torch.pow(d_inv, 0.5))
    x = feature.to(args.device)
    feature = feature.to(args.device)
    graph = graph.to(args.device)
    graph = graph.add_self_loop()
    
    W = compute_heat(adj, t=args.t).to(args.device)
    
    Lap1 = torch.eye(N, dtype=torch.float) - torch.mm(torch.mm(D_inv_1_2, adj), D_inv_1_2)
    Lap2 = Lap1 @ Lap1  
    Laps = torch.stack([I, Lap1, Lap2], dim=2).to(args.device)
 
    
    print("Start Traing...")
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        optimizer.zero_grad()
                       
        h, z = model(graph, feature)
        
        c = torch.mm(torch.mm(z.T, -W), z) / N / args.out_dim
        loss_stru = torch.diagonal(c).sum()
        
        x_hat = model.reconstruct(Laps, h) 
        loss_fea = inner_loss(x, x_hat, 1)   
        
        loss = loss_fea * args.lambd + loss_stru * args.beta
        
        loss.backward()
        optimizer.step()
               
    
    print("Evaluating...")
    graph = graph.to(args.device)
    graph = graph.remove_self_loop().add_self_loop()
    feature = feature.to(args.device)

    embeds = model.get_embedding(graph, feature)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]
    
    
    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_classes)
    opt = torch.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                eval_acc = test_acc


    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    print("parameters:", sum(param.numel() for param in model.parameters()))
    
    
