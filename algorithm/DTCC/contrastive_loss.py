import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class InstanceLoss(nn.Module):
    """
    Instance-level loss
    """
    def __init__(self, batch_size, temperature, device,k_num):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2)
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.k_num = k_num

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j,pseudo_label,epoch,epoch_limit):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim1 = torch.matmul(z_i, z_j.T) / self.temperature
        if epoch > epoch_limit:
            position_tensor = (pseudo_label.unsqueeze(0) == pseudo_label.unsqueeze(1)).int()
            position_tensor = position_tensor.to(self.device)
            brother = self.near(sim1 * position_tensor, self.k_num - 1)
        else:
            brother = self.near(sim1, self.k_num - 1)
        one = torch.eye(self.batch_size, dtype=torch.int32).to(self.device)
        brother_new = torch.max(brother, one)
        zero_matrix = torch.zeros((self.batch_size, self.batch_size)).to(self.device)
        w1 = torch.cat((zero_matrix, brother_new), 1)
        w2 = torch.cat((brother_new, zero_matrix), 1)
        w_final = torch.cat((w1, w2), 0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim[sim == 0] = 1e-45
        w_sim = w_final * sim
        positive_samples = w_sim[w_sim > 0].reshape(N, -1)
        negative_samples = w_sim[w_sim == 0].reshape(N, -1)
        negative_samples = negative_samples[:,:-1]
        labels = torch.zeros(N, N - 1).to(positive_samples.device).long()

        labels[:, 0] = 1.0
        labels[:, :self.k_num] = 1.0
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels.float())
        loss /= N * (self.k_num )
        return loss

    def near(self,x,k_num):

        x.fill_diagonal_(-999)
        indices = torch.topk(x, k_num, dim=1).indices 

        y = torch.zeros_like(x, dtype=torch.float32)

        y.scatter_(1, indices, 1) 

        return y
class ClusterLoss(nn.Module):
    """
    Cluster-level loss
    """
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        x = c.unsqueeze(1)
        y = c.unsqueeze(0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + ne_loss
