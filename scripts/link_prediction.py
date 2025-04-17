import argparse
import copy
import random
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

import sys
import os 
sys.path.insert(0, os.getcwd())
import time

from datasets.data_util import create_link_prediction_dataset   
from utils.from_numpy import FromNumpy

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")


class LinkPrediction(object):
    """Link prediction task"""
    
    def __init__(self, dataset, hidden_size, num_shuffle, seed, batch_size, gpu, secret_dataset, emb_path):
        self.data = create_link_prediction_dataset(dataset)
        self.secret_data = create_link_prediction_dataset(secret_dataset) if secret_dataset != "" else None
        self.edges = self.data.edges
        self.edge_labels = self.data.edge_labels
        if self.secret_data is not None:
            self.edges_secret = self.secret_data.edges
            self.edge_labels_secret = self.secret_data.edge_labels
        
        self.model = FromNumpy(hidden_size, emb_path)

        self.hidden_size = hidden_size
        self.num_shuffle = num_shuffle
        self.seed = seed
        
        self.gpu = gpu
        self.batch_size = batch_size
        
    def train(self):
        G = nx.Graph()
        G.add_edges_from(self.data.data.edge_index.t().tolist())
        embeddings = self.model.train()
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        if self.secret_data is None:
            return self._evaluate(self.edges, self.edge_labels, embeddings,
                                  self.num_shuffle, self.batch_size, self.gpu)
        else:
            embeddings_secret = embeddings
            return self._evaluate_secret(self.edges, self.edge_labels, embeddings,
                                         self.edges_secret, self.edge_labels_secret, embeddings_secret,
                                         self.num_shuffle, self.batch_size, self.gpu)
        
    def _evaluate(self, edges, edge_labels, embeddings, num_shuffle, batch_size, gpu):
        # shuffle, to create train/test groups
        skf = StratifiedKFold(n_splits=num_shuffle, shuffle=True, random_state=self.seed)
        idx_list = []
        
        for idx in skf.split(np.zeros(len(edge_labels)), edge_labels):
            idx_list.append(idx)
        
        all_results = defaultdict(list)
        
        for train_idx, test_idx in idx_list[:2]:
            edges_train = edges[train_idx]
            y_train = edge_labels[train_idx]
            
            edges_test = edges[test_idx]
            y_test = edge_labels[test_idx]
            
            train_dataset = TensorDataset(edges_train, y_train)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            
            test_dataset = TensorDataset(edges_test)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
            
            link_predictor = Link_Predictor(embeddings, 32)
            trainer = pl.Trainer(
                gpus=gpu,
                max_epochs=50,
            )
            
            
            trainer.fit(model=link_predictor,
                        train_dataloaders=train_dataloader,
                        )
            
            
            preds = trainer.predict(link_predictor, test_dataloader)
            preds = torch.cat(preds)
            auc = roc_auc_score(y_test, preds)
            all_results["AUC"].append(auc)
            
            preds = torch.where(preds > 0.5, 1, 0)
            f1 = f1_score(y_test, preds, average='micro')
            all_results["Micro-F1"].append(f1)
            
        return dict(
                    (
                        f"{train_percent}", "{:.5} ({:.5})".format(
                            sum(all_results[train_percent]) / len(all_results[train_percent]),
                            np.std(all_results[train_percent]))
                    )
                    for train_percent in sorted(all_results.keys())
                )
        
    def _evaluate_secret(self, edges, edge_labels, embeddings, edges_secret, edge_labels_secret, embeddings_secret, num_shuffle, batch_size, gpu):
        # shuffle, to create train/test groups
        skf = StratifiedKFold(n_splits=num_shuffle, shuffle=True, random_state=self.seed)
        idx_list = []
        
        for idx in skf.split(np.zeros(len(edge_labels)), edge_labels):
            idx_list.append(idx)
        
        all_results = defaultdict(list)
        
        for train_idx, test_idx in idx_list:
            edges_train = edges[train_idx]
            y_train = edge_labels[train_idx]
            
            edges_test = edges_secret
            y_test = edge_labels_secret
            
            train_dataset = TensorDataset(edges_train, y_train)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            
            test_dataset = TensorDataset(edges_test)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
            
            link_predictor = Link_Predictor(embeddings, 32)
            link_predictor_secret = Link_Predictor(embeddings_secret, 32)
            trainer = pl.Trainer(
                gpus=gpu,
                max_epochs=50,
            )
            
            trainer.fit(model=link_predictor,
                        train_dataloaders=train_dataloader)
            
            link_predictor_secret.encoder.load_state_dict(link_predictor.encoder.state_dict())

            preds = trainer.predict(link_predictor_secret, test_dataloader)
            preds = torch.cat(preds)
            
            auc = roc_auc_score(y_test, preds)
            all_results["AUC"].append(auc)
            
            preds = torch.where(preds > 0.5, 1, 0)
            f1 = f1_score(y_test, preds, average='micro')
            all_results["Micro-F1"].append(f1)
            
        return dict(
                    (
                        f"{train_percent}", "{:.5} ({:.5})".format(
                            sum(all_results[train_percent]) / len(all_results[train_percent]),
                            np.std(all_results[train_percent]))
                    )
                    for train_percent in sorted(all_results.keys())
                )
            
class Link_Predictor(pl.LightningModule):
    def __init__(self, embeddings, hidden_dim):
        super().__init__()
        node_num, emb_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(node_num, emb_dim,)
        self.embedding_layer.weight = nn.Parameter(embeddings)
        self.embedding_layer.weight.requires_grad = False
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim,),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, edges):
        embs = self.embedding_layer(edges)
        src_node_embs = embs[:, 0, :]
        tgt_node_embs = embs[:, 1, :]
        edge_embs = torch.cat([src_node_embs, tgt_node_embs], dim=1)
        edge_embs = self.encoder(edge_embs).flatten()

        preds = torch.sigmoid(edge_embs)
        return preds
    
    def training_step(self, batch, batch_idx):
        edges, y = batch
        y_hat = self(edges)
        loss = -1 * (y * torch.log(y_hat + 1e-10) + (1-y) * torch.log(1-y_hat + 1e-10)).mean()
        return loss
    
    def validation_step(self, batch, batch_idx):
        edges, y = batch
        y_hat = self(edges)
        val_loss = -1 * (y * torch.log(y_hat + 1e-10) + (1-y) * torch.log(1-y_hat + 1e-10)).mean()
        # val_loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        
    def predict_step(self, batch, batch_idx):
        edges = batch[0]
        y_hat = self(edges)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return optimizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--secret_dataset", type=str, default="")
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffle", type=int, default=10)
    parser.add_argument("--emb-path", type=str, default="")
    parser.add_argument("--batch_size", default=512, type=int,)
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")
    args = parser.parse_args()
    
    
    task = LinkPrediction(
        args.dataset,
        args.hidden_size, 
        args.num_shuffle,
        args.seed,
        batch_size=args.batch_size,
        gpu=args.gpu,
        emb_path=args.emb_path,
        secret_dataset=args.secret_dataset
    )
    
    ret = task.train()
    print(ret)
    
    save_path = args.emb_path.split('/')[1]
    save_dir = os.path.join('link_prediction_results', args.emb_path.split("/")[0])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    file_name = 'link_prediction_' + args.emb_path.split('/')[1] + ".txt"
    
    with open(os.path.join(save_dir, file_name), 'a') as f:
        f.write(str(ret) + "\n")
    