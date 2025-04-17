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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.getcwd())
from utils.from_numpy import FromNumpy 
sys.path.insert(0, os.getcwd())
import time

from datasets.data_util import create_node_classification_dataset
from utils.from_numpy import FromNumpy

warnings.filterwarnings("ignore")


class NodeClassification(object):
    """Node classification task."""

    def __init__(self, dataset, hidden_size, num_shuffle, seed, batch_size, gpu, emb_path):
        dataset_infos = create_node_classification_dataset(dataset)
        self.data = dataset_infos.data
        self.node2id = dataset_infos.node2id
        self.label_matrix = self.data.y
        self.num_nodes, self.num_classes = self.data.y.shape

        self.model = FromNumpy(hidden_size, emb_path)
        self.hidden_size = hidden_size
        self.num_shuffle = num_shuffle
        self.seed = seed
        
        # lr training parameters
        self.gpu = gpu
        self.batch_size = batch_size
        

    def train(self):
        G = nx.Graph()
        G.add_nodes_from(self.node2id.values())
        embeddings = self.model.train()

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]

        # remove_nodes_without_labels
        label_matrix = torch.Tensor(self.label_matrix)
        id_with_label_flags = label_matrix.sum(-1) > 0
        label_matrix = label_matrix[id_with_label_flags]
        features_matrix = features_matrix[id_with_label_flags]

        return self._evaluate(features_matrix, label_matrix, self.num_shuffle, self.batch_size, self.gpu)

        
    def _evaluate(self, features_matrix, label_matrix, num_shuffle, batch_size, gpu):
        input_dim, output_classes = features_matrix.shape[-1], label_matrix.shape[-1]
        # shuffle, to create train/test groups
        skf = StratifiedKFold(n_splits=num_shuffle, shuffle=True, random_state=self.seed)
        idx_list = []
        labels = label_matrix.argmax(axis=1).squeeze().tolist()
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)

        # score each train/test group
        all_results = defaultdict(list)

        for train_idx, test_idx in idx_list:

            X_train = torch.tensor(features_matrix[train_idx], dtype=torch.float32)
            y_train = torch.tensor(label_matrix[train_idx], dtype=torch.float32)

            X_test = torch.tensor(features_matrix[test_idx], dtype=torch.float32)
            y_test = torch.tensor(label_matrix[test_idx], dtype=torch.float32)


            X_train_dataset = TensorDataset(X_train, y_train)
            X_train_dataloader = DataLoader(X_train_dataset, shuffle=True, batch_size=batch_size)
            lr = Logistic_Classifier(input_dim, output_classes)
            trainer = pl.Trainer(
                    gpus=gpu,
                    max_epochs=100,
            )

            trainer.fit(
                model = lr,
                train_dataloaders=X_train_dataloader,
            )

            lr.eval()
            preds = lr(X_test).argmax(-1)

            result = f1_score(y_test.argmax(-1), preds, average="micro")

            all_results[""].append(result)

        return  dict(
                    (
                        f"{train_percent}", "{:.5} ({:.5})".format(
                            sum(all_results[train_percent]) / len(all_results[train_percent]),
                            np.std(all_results[train_percent]))
                    )
                    for train_percent in sorted(all_results.keys())
                )



class Logistic_Classifier(pl.LightningModule):
    def __init__(self, input_dim, output_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_classes)

    def forward(self, x):
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        x = batch
        pred = self(x).argmax(-1)
        return pred
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return optimizer




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffle", type=int, default=10)
    parser.add_argument("--emb-path", type=str, default="")
    parser.add_argument("--batch_size", default=512, type=int,)
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")
    args = parser.parse_args()

    task = NodeClassification(
        args.dataset,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        batch_size=args.batch_size,
        gpu=args.gpu,
        emb_path=args.emb_path,
    )

    ret = task.train()
    print(ret)

    # pretrain_model_name = args.emb_path.split('/')[1]
    # time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    # with open(r'./experiments/records/{}_{}_{}.txt'.format(args.dataset, pretrain_model_name, time_now), 'w') as f:
    #     f.write('Micro-F1: {:.5f}\n'.format(ret["Micro-F1"]))