import itertools
import os
import random
import time
import math
import pandas as pd
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from algorithm.DTCC.DTCC import DTCC
from algorithm.DTCC.dataset import Load_Dataset, MyDataset
from torch.utils.data import DataLoader
from algorithm.DTCC import contrastive_loss
from algorithm.DTCC.seed import set_seed
import numpy as np

from sklearn.manifold import TSNE
# For the UCI ML handwritten digits dataset
from sklearn.datasets import load_digits

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns


class model():
    def __init__(self):
        # optimizer parameters
        self.k_num = 1
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0003
        self.weight_decay = 0.000000001

        # ------freq_encoder parameters------
        self.input_channels = 1  # The number of input channels of the convolutional network with a UTS of 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 64  # The number of convolutional network output channels
        self.num_classes = None
        self.dropout = 0.30
        # --------------------------------
        self.epochs = 300
        self.model_save_path = None
        self.tensorboard_path = None
        # contrastive_loss parameters
        self.instance_temperature = 0.5
        self.cluster_temperature = 1.0
        self.lam = 0.7  # Loss function coefficient
        self.lamc = 1 # Loss function coefficient
        # device parameters
        self.device = 'cuda'

        # DataLoader parameters
        self.batch_size = 128
        self.drop_last = True
        self.num_workers = 0

        # Time augmentations parameters
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8

        # Frequency augmentations parameters
        self.remove_frequency_ratio = 0.1
        self.add_frequency_ratio = 0.1

        # Parameters for the instance-level and cluster-level mapping networks
        self.CNNoutput_channel = None
        self.feature_dim = 256
        self.hidden_size = 1024
        self.output_size = 512

        self.dropout_rate = 0.10
        self.num_layers = 2  # The number of layers of BiLSTM
        self.k_num = 1
        self.epoch_limit = 200

    def step_epoch(self, optimizer, dataset, criterion_instance, criterion_cluster, epoch):  # 每一次迭代
        loss_epoch = 0
        total_loss = []
        for step, (x_data, y_data, aug1, aug2, x_data_f, aug1_f, aug2_f) in enumerate(dataset):
            optimizer.zero_grad()

            x_data = x_data.to(torch.float32).to(self.device)
            x_data_f = x_data_f.to(torch.float32).to(self.device)

            aug1 = aug1.to(torch.float32).to(self.device)
            aug1_f = aug1_f.to(torch.float32).to(self.device)

            aug2 = aug2.to(torch.float32).to(self.device)
            aug2_f = aug2_f.to(torch.float32).to(self.device)

            """Representation"""
            h_t, z_i_t, z_c_t, h_t_aug, z_i_t_aug, z_c_t_aug = self.model(aug1, aug2, 't')
            h_f, z_i_f, z_c_f, h_f_aug, z_i_f_aug, z_c_f_aug = self.model(aug1_f, aug2_f, 'f')
            pseudo_label = torch.argmax((z_c_t_aug + z_c_f_aug) / 2, dim=1)

            # Time domain contrastive constraints
            loss_i_t = criterion_instance(z_c_t, z_c_t_aug, pseudo_label, epoch, self.epoch_limit)
            loss_c_t = criterion_cluster(z_c_t, z_c_t_aug)
            loss_t = self.lamc * loss_i_t + loss_c_t

            # Frequency domain contrastive constraints
            loss_i_f = criterion_instance(z_c_f, z_c_f_aug, pseudo_label, epoch, self.epoch_limit)
            loss_c_f = criterion_cluster(z_c_f, z_c_f_aug)
            loss_f = self.lamc * loss_i_f + loss_c_f

            z_aug = (z_c_t_aug + z_c_f_aug) / 2
            # Cross-domain contrastive constraints
            loss_i_t_f1 = criterion_instance(z_aug, z_c_f_aug, pseudo_label, epoch, self.epoch_limit)
            loss_c_t_f1 = criterion_cluster(z_aug, z_c_f_aug)
            loss_i_t_f2 = criterion_instance(z_aug, z_c_t_aug, pseudo_label, epoch, self.epoch_limit)
            loss_c_t_f2 = criterion_cluster(z_aug, z_c_t_aug)

            loss_tf = self.lamc * loss_i_t_f1 + loss_c_t_f1 + self.lamc * loss_i_t_f2 + loss_c_t_f2
            # Loss Function
            loss = self.lam * (loss_t + loss_f) + (1 - self.lam) * loss_tf
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            loss_epoch += loss.item()
        total_loss = torch.tensor(total_loss).mean()
        return total_loss.item()

    def train(self, ds, valid_ds=None, valid_func=None, cb_progress=lambda x: None):
        model_path = os.path.dirname(__file__)
        self.class_num = len(np.unique(ds[1]))
        self.input_channels = ds[0].shape[1]
        self.input_size = ds[0].shape[2]
        max_result = 0
        max_record = []
        set_seed()
        max_result = 0
        max_record = []
        trainset = Load_Dataset(self, ds)
        # if self.batch_size == 'all':
        # self.batch_size = trainset.len
        test_set = MyDataset(ds)
        # Make sure that the dimensions of your data are [num_instance,in_channel,series_length
        self.model = DTCC(self).to(self.device)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.batch_size, shuffle=True,
                                                    num_workers=self.num_workers, drop_last=self.drop_last)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False,
                                                    num_workers=self.num_workers, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                        weight_decay=self.weight_decay)
        criterion_instance = contrastive_loss.InstanceLoss(self.batch_size,
                                                            self.instance_temperature,
                                                            self.device, math.floor(
                self.k_num_lam )).to(self.device)
        criterion_cluster = contrastive_loss.ClusterLoss(self.class_num,
                                                            self.cluster_temperature,
                                                            self.device).to(self.device)
        results_df = pd.DataFrame(columns=['Epoch', 'NMI', 'RI'])
        start_time = time.time()
        cumulative_time = start_time
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loss_epoch = self.step_epoch(optimizer, train_loader, criterion_instance, criterion_cluster, epoch)
            predict_labels, true_label, clu_time, clu_fre = self.predict_epoch(test_loader)
            # Adjust the learning rate
            adjust_learning_rate(optimizer, self.lr, epoch, self.epochs)
            result = [e(predict_labels, true_label) for e in valid_func]
            valid_f = [str(v) for v in valid_func]
            if max_result < result[0]:
                max_result = result[0]
                max_record = result
                print(result)

        print(self.model_save_path + '   NMI   ' + str(max_record[0]) + '   RI    ' + str(max_record[1]))
        self.pred_labels = predict_labels
        return train_loader

    def predict_epoch(self, test_loader):
        self.model.eval()
        feature_vector = []
        labels_vector = []
        clu_vector = []
        clu_vefre = []
        for step, (x_data, y_data) in enumerate(test_loader):
            x = x_data.to(torch.float32).to(self.device)
            with torch.no_grad():
                c, clu_time, clu_fre = self.model.forward_cluster(x)
            c = c.detach()
            clu_time = clu_time.detach()
            clu_fre = clu_fre.detach()
            feature_vector.extend(c.cpu().detach().numpy())
            clu_vector.extend(clu_time.cpu().detach().numpy())
            clu_vefre.extend(clu_fre.cpu().detach().numpy())
            labels_vector.extend(y_data)
        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        return feature_vector, labels_vector, clu_vector, clu_vefre

    def predict(self, ds, cb_progress=lambda x: None):
        return self.pred_labels  # Take the clustering results of the last epoch


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def list_size(ds):
    x = ds[0].shape[0]
    min_power_index = math.ceil(math.log2(x / 4))
    min_power = 2 ** min_power_index

    max_power_index = math.floor(math.log2(x))
    powers_of_two = [2 ** i for i in range(min_power_index, max_power_index + 1) if 2 ** i <= x]
    return powers_of_two[-2:] if len(powers_of_two) > 1 else powers_of_two
