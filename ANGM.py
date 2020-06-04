'''
author: liuxueyan
project: ANGM
update: gamma, pi, omega, gmu and gvar are updated by variational inference, 
        others are updated by back propagation (gradient descent or its variation)
'''
from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image 
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.autograd import Variable
import scipy.sparse as sp
import time

import gzip
import sys
import pickle
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from collections import Counter
import pandas as pd

import numpy as np


from matplotlib import pylab as pl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


filename = 'cornell'

edges = np.loadtxt(filename + '.edges')
N = int(np.max(edges)) + 1
adj = sp.coo_matrix((np.ones(edges.shape[0]), (np.array(list(edges[:,0])),np.array(list(edges[:,1])))),shape=(N, N), dtype=np.float32)
A = adj.todense()
adj = torch.tensor(A)



parser = argparse.ArgumentParser(description='ANGM ')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--K', type=int, default=1, metavar='K',
                    help='the number of cluasters (default: 10)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='D',
                    help='the dimention of embedding (default: 32)')
parser.add_argument('--layer', type=int, default=2, metavar='D',
                    help='the layer of NN (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='D',
                    help='the lr of grad (default: 1e-2)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torpi = torch.Tensor([np.pi])
device = torch.device("cuda" if args.cuda else "cpu")
torpi = torpi.to(device)
adj = adj.to(device)


class DealDataset(Dataset):

    def __init__(self):
              
        with open(filename + '.attributes') as f:
            data_list = f.readlines()
            data = []
            for i in data_list:
                i = i.strip()
                tmp = i.split(' ')
                data.append(tmp)
            attri = []
            for i in data:
                data1 = []
                for j in i:
                    data1.append(float(j))
                attri.append(data1)

        with open(filename + '.label') as f:
            data_list = f.readlines()
            data = []
            for i in data_list:
                i = i.strip()
                data.append(i)
            label = []
            for j in data:
                label.append(float(j))
        
            data_list = [i.split('\n')[0] for i in data_list]
            data_list = [i.split(' ') for i in data_list]

        self.X = np.array(attri, dtype = np.float32)
        self.Y = np.array(label, dtype = np.int32)
        
        self.len = self.X.shape[0]
        self.attrlen = self.X.shape[1]
          
        self.K = np.max(self.Y) + 1
        pass

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len

    
dealDataset = DealDataset()

train_loader = DataLoader(dataset=dealDataset,
                          batch_size=int(dealDataset.len),
                          shuffle=False)


test_loader = DataLoader(dataset=dealDataset,
                          batch_size=int(dealDataset.len),
                          shuffle=False)



kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class ANGM(nn.ModuleList):
    def __init__(self, inter_dim, latent_dim, K, adj, N, attr_len):
        super(ANGM, self).__init__()
        self.K = K
        self.N = N
        self.adj = adj
        self.latent_dim = latent_dim
        self.adj1 = self.adj-self.adj.diag().diag()
        self.adj2 = 1-self.adj1-(1-self.adj1).diag().diag()
        self.fc1 = nn.Linear(attr_len, inter_dim[0])
        self.fc21 = nn.Linear(inter_dim[1], latent_dim)
        self.fc22 = nn.Linear(inter_dim[1], latent_dim)
        self.fc3 = nn.Linear(latent_dim, inter_dim[-1])
        self.fc4 = nn.Linear(inter_dim[-2], attr_len)

        
        
        self.bpi = torch.rand([K,K])/K # link probobality between blocks
        self.bpi = self.bpi.to(device)
        self.omega = torch.ones([K])/K
        self.omega = self.omega.to(device)
        gamma = torch.rand([N,K])/K # proximate posterior of C
        self.gamma = gamma/torch.sum(gamma, dim=1, keepdim=True)
        self.gamma = self.gamma.to(device)
        self.initial()
 

        
    def xarv_init(self, tensor):
        nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('relu'))
    def initial(self):
        for w in [self.fc1,  self.fc21, self.fc22, self.fc3, self.fc4]:
            self.xarv_init(w.weight)
 
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self):
        std = torch.exp(0.5*self.logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(self.mu)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))


   
    def forward(self, x):
        mu, logvar = self.encode(x)
        self.mu = mu
        self.logvar = logvar
        z = self.reparameterize()
        return z, self.decode(z)

    def initial_gmm(self, x):
        with torch.no_grad():
            z, _ = self.forward(x)
        gmm = mixture.GaussianMixture(n_components=self.K, covariance_type='diag').fit(z.cpu())
        self.gmu = torch.tensor(gmm.means_).to(device).t().float()
        self.loggvar = torch.log(torch.tensor(gmm.covariances_+1e-10)).to(device).t().float()



    def get_gamma(self):
        temp_u = self.gmu.unsqueeze(0)
        temp_gvar = self.loggvar.unsqueeze(0)
        u_t = self.mu.unsqueeze(2)
        logvar_t = self.logvar.unsqueeze(2)

        temp_omega = self.omega.unsqueeze(0)

        term1 = self.adj1.mm(self.gamma).mm(torch.log(self.bpi.t()+1e-10)) + self.adj2.mm(self.gamma).mm(torch.log(1-self.bpi.t()))
        term2 = -torch.sum(temp_gvar  + logvar_t.exp()/(temp_gvar.exp()) + (u_t - temp_u).pow(2)/(temp_gvar.exp()),1)

        log_p_z =  term1+ 0.5*term2 + torch.log(temp_omega+1e-10)
        maxlogpz, _ = torch.max(log_p_z,1)
        log_p_z_t = log_p_z - maxlogpz.unsqueeze(1)
        p_z = torch.exp(log_p_z_t)

        self.gamma = p_z/(torch.sum(p_z, dim = 1, keepdim=True) +1e-10)
        pass

    def get_pi(self):
        block_node = torch.sum(self.gamma,dim = 0)
        self.bpi = self.gamma.t().mm(self.adj).mm(self.gamma)/(block_node.unsqueeze(1).mm(block_node.unsqueeze(0)) + 1e-3)
    

    def get_omega(self):
        self.omega = torch.sum(self.gamma,0)/self.N

    def get_gmu(self):
        block_node = torch.sum(self.gamma,dim = 0)
        self.gmu = self.mu.t().mm(self.gamma)/(block_node.unsqueeze(0)+1e-10)
        pass


    def get_loggvar(self):
        block_node = torch.sum(self.gamma,dim = 0)
        temp_u = self.gmu.unsqueeze(0)
        u_t = self.mu.unsqueeze(2)
        logvar_t = self.logvar.unsqueeze(2)
        gamma_t = self.gamma.unsqueeze(1)
        
        gvar = torch.sum(gamma_t*(logvar_t.exp()+(u_t - temp_u).pow(2)),0)/(block_node.unsqueeze(0)+1e-10)
        self.loggvar = torch.log(gvar + 1e-10)
        pass
        

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x):   

        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        u_t = self.mu.unsqueeze(2)
        logvar_t = self.logvar.unsqueeze(2)
    
        self.get_gamma()
        self.gamma = self.gamma.detach()
        gamma_t = self.gamma.unsqueeze(1)
    
        self.get_pi()
        self.bpi = self.bpi.detach()
    
        self.get_omega()
        self.omega = self.omega.detach()
    
        self.get_gmu()
        self.gmu = self.gmu.detach()

        self.get_loggvar()
        self.loggvar = self.loggvar.detach()

        temp_u = self.gmu.unsqueeze(0)
        temp_gvar = self.loggvar.unsqueeze(0)
        temp_omega = self.omega.unsqueeze(0)

   
        lossterm1 = -torch.sum(self.gamma.mm(torch.log(self.bpi+1e-10)).mm(self.gamma.t())*self.adj1 + self.gamma.mm(torch.log((1 - self.bpi)+1e-10)).mm(self.gamma.t())*self.adj2)
        lossterm2 = BCE
        lossterm3 = 0.5*torch.sum(gamma_t*torch.log(2*torpi))   + torch.sum(0.5*gamma_t*(temp_gvar \
               + logvar_t.exp()/(temp_gvar.exp()) + (u_t - temp_u).pow(2)/(temp_gvar.exp())))
        lossterm4 = torch.sum(self.gamma*torch.log(self.gamma+1e-10)) - torch.sum(self.gamma*torch.log(temp_omega+1e-10))
        lossterm5 =  - torch.sum(0.5*(1+self.logvar))
        loss = lossterm1 + lossterm2 + lossterm3 +lossterm4 + lossterm5
    
        return loss
     


inter_dim = [32,32,32]
model = ANGM(inter_dim, args.latent_dim, dealDataset.K, adj,N, dealDataset.attrlen).to(device)

model.initial_gmm(torch.tensor(train_loader.dataset.X).to(device))



optimizer = optim.Adam(model.parameters(), lr=args.lr)


def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.shape[0] == Y.shape[0]
  Y_pred = Y_pred.type(Y.dtype)
  D = torch.max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.shape[0]):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.shape[0],ind

def train(epoch, K):
    model.train()
    train_loss = 0
    train_acc = 0
    for _, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        z, recon_batch = model(data)
        loss = model.loss_function(recon_batch, data)
        

        z_tmp = z.detach()
            
        try:
            gmm = mixture.GaussianMixture(n_components=K, covariance_type='diag').fit(z_tmp.to('cpu').numpy())
            GMM_pred = torch.Tensor(gmm.predict(z_tmp.to('cpu')))
        except Exception:
            gm_train_nmi = 0.0
            gm_train_accuracy = 0.0
        else:
            gm_train_nmi = nmi(label.to('cpu'), GMM_pred)
            gm_train_accuracy, _ = cluster_acc(GMM_pred, label.to('cpu'))

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return z, label, train_loss, gm_train_accuracy, gm_train_nmi

if __name__ == "__main__":
    
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        z, label, train_loss, gm_train_accuracy, gm_train_nmi = train(epoch, dealDataset.K)
        
        
        print('====> Epoch: {} Average loss: {:.4f}   GMMTrain acc: {:.4f}, GMMTrain nmi: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset),  gm_train_accuracy, gm_train_nmi))
        
     

    
    
