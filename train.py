from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, read_text_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=float, default=0.0, help='Alpha for the leaky_relu.')
parser.add_argument('--nb_heads', type=int, default=1 , help='Number of head attentions.')
parser.add_argument('--dataset-str', type=str, default='course', help='type of dataset.')

args = parser.parse_args()

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def gae_for(args, seed_num):
    np.random.seed(seed_num)
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    t_adj, t_features, tfidf_feature = read_text_data(args.dataset_str) #文本特征与矩阵
    n_nodes, feat_dim = features.shape
    t_n_nodes, t_feat_dim = t_features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    t_adj_norm = preprocess_graph(t_adj) #进行归一化处理
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.DoubleTensor(np.array(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()))
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, t_feat_dim, t_n_nodes, args.hidden1, args.hidden2, args.dropout, args.alpha, args.nb_heads)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        
        recovered, mu, logvar = model(features, adj_norm, t_features, t_adj_norm, tfidf_feature)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        train_acc = get_acc(recovered,adj_label)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "train_acc=", "{:.5f}".format(train_acc),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    roc_score, ap_score, preds_all, labels_all = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    return roc_score, ap_score, preds_all, labels_all

if __name__ == '__main__':
    
    roc_score, ap_score, preds_all, labels_all = gae_for(args, args.seed)
    
    preds_all_temp  = preds_all
    preds_all_temp[preds_all_temp >= 0.6] = 1
    preds_all_temp[preds_all_temp < 0.6] = 0
    
    ACC = accuracy_score(labels_all, preds_all_temp)
    F1 = f1_score(labels_all, preds_all_temp)
    
    print('Test acc score: ', float('%.4f' %ACC))
    print('Test f1 score: ', float('%.4f' %F1))
    print('Test map_score score: ', float('%.4f' %ap_score))
    print('Test auc score: ', float('%.4f' %roc_score))        
    
    
    