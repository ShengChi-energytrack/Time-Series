
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
import pandas as pd
import numpy as np

import mrnn_utility as mrnn_utility
from mrnn_utility import block_diag,unblock_diag, ReLU



# Use cuda (GPU) to train models if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
else:
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float32)

class interpolater(nn.Module):
    '''Interpolater Module for the M-RNN model.'''
    def __init__(self,nchannels,hidden_dim,seq_len,padding="replication",act='relu'):
        super(interpolater, self).__init__()
        self.nchannels = nchannels
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Activation function
        if act == 'relu':
            self.act = F.relu
        elif act == 'tanh':
            self.act = F.tanh
        
        # Forward RNN weights
        self.Wf = Parameter(block_diag(xavier_uniform_(torch.FloatTensor(nchannels,hidden_dim,hidden_dim).to(device=device))))
        self.Vf = Parameter(block_diag(xavier_uniform_(torch.FloatTensor(nchannels,hidden_dim,3).to(device=device))))
        self.cf = Parameter(block_diag(xavier_uniform_(torch.FloatTensor(nchannels,hidden_dim,1).to(device=device))))

        # Backward RNN weights
        self.Wb = Parameter(block_diag(xavier_uniform_(torch.FloatTensor(nchannels,hidden_dim,hidden_dim).to(device=device))))
        self.Vb = Parameter(block_diag(xavier_uniform_(torch.FloatTensor(nchannels,hidden_dim,3).to(device=device))))
        self.cb = Parameter(block_diag(xavier_uniform_(torch.FloatTensor(nchannels,hidden_dim,1).to(device=device))))

        # Hidden state weights (combine forward and backward)
        self.U = Parameter(block_diag(xavier_uniform_(torch.FloatTensor(nchannels,1,hidden_dim*2).to(device=device))))
        self.c0 = Parameter(block_diag(nn.init.normal_(torch.FloatTensor(nchannels,1,1),std=.1).to(device=device)))
        
        # Padding options
        if padding == "replication":
            self.pad = nn.ReplicationPad2d((1,1,0,0))
        elif padding == "zero":
            self.pad = nn.ZeroPad2d((1,1,0,0))
            
    def forward(self,x,m,d):
        '''A forward pass through the interpolater. This function will go through all time steps.
        Input:
        x = Measurement
        m = Mask (1=observed, 0=missing)
        d = time elapsed since last observation
        
        Output:
        Estimate x_est for every input time step.'''
        
        batchsize = x.shape[0]
        # Initialize hidden states
        hidden_forwards = [torch.zeros(batchsize,self.hidden_dim*self.nchannels,self.nchannels)]
        hidden_backwards = [torch.zeros(batchsize,self.hidden_dim*self.nchannels,self.nchannels)]
        
        # Append zeros to beginning and end of input
        x = self.pad(x.unsqueeze(0)).squeeze(0)
        m = self.pad(m.unsqueeze(0)).squeeze(0)
        d = self.pad(d.unsqueeze(0)).squeeze(0)
        
        # Iterate through time (backward and forward)
        for t in range(self.seq_len):  
            # forward RNN hidden states
            hidden_f = self.act(torch.matmul(self.Wf,hidden_forwards[t]) + torch.matmul(self.Vf,
                block_diag(torch.stack((x[:,:,t],m[:,:,t],d[:,:,t]),axis=2).view(-1,self.nchannels,3,1))) 
                +self.cf)
            
            # backward RNN hidden states
            hidden_b = self.act(torch.matmul(self.Wb,hidden_backwards[t]) + 
                  torch.matmul(self.Vb,block_diag(torch.stack((x[:,:,self.seq_len+1-t],m[:,:,self.seq_len+1-t],
                  d[:,:,self.seq_len+1-t]),axis=2).view(-1,self.nchannels,3,1))) 
                  + self.cb)
            
            hidden_forwards.append(hidden_f)
            hidden_backwards.append(hidden_b)

        hidden_forwards = hidden_forwards[1:] # delete state t=-1
        hidden_backwards = hidden_backwards[1:][::-1] # delete state t=T+1 and reverse the list
        
        final_hidden=torch.empty(batchsize,self.nchannels,self.seq_len)
        
        # Iterate through time again and compute combined state
        for t in range(self.seq_len):
            hidden = self.act(torch.matmul(self.U,block_diag(torch.cat((
                unblock_diag(hidden_forwards[t],n=self.nchannels,size_block=(self.hidden_dim,1)),
                unblock_diag(hidden_backwards[t],n=self.nchannels,size_block=(self.hidden_dim,1))),axis=2))) 
                + self.c0)
            
            final_hidden[:,:,t]= unblock_diag(hidden,n=self.nchannels).flatten(1)
            
        return final_hidden
    
class imputer(nn.Module):
    '''Fully connected network that computes the imputation across data streams. We can use the time dimension
    as the batch dimension here, as the linear layers are independent of time.'''
    def __init__(self,n_channels,hidden_dim=3,act='relu'):
        super(imputer, self).__init__()
        if act == 'relu':
            self.act = F.relu
        elif act == 'tanh':
            self.act = F.tanh
        self.V1 = nn.Linear(n_channels,hidden_dim,bias=False)
        self.V2 = nn.Linear(n_channels,hidden_dim,bias=False)
        self.U = nn.Linear(n_channels,hidden_dim) # bias beta
        self.W = nn.Linear(hidden_dim,n_channels) # bias alpha
        
    def forward(self,x,x_est,m):
        '''x : true measurement
        x_est : estimated measurement of the interpolater
        m : mask'''
        v1out = self.V1(x_est.permute(0,2,1))
        v2out = self.V2(m.permute(0,2,1))
        self.U.weight.data.fill_diagonal_(0) # diagonal to zero to prevent usage of x_t^d at for prediction x_hat_t^d
        uout = self.U(x.permute(0,2,1)) 
        h = self.act(uout+v1out+v2out) # hidden layer
        out = self.W(h) # output layer, linear activation here
        return out.permute(0,2,1)
    
class MRNN(nn.Module):
    def __init__(self,nchannels,seq_len,hidden_dim_inter,hidden_dim_imp=3,verbose=False,padding="replication",act='relu'):
        super(MRNN, self).__init__()
        self.inter = interpolater(nchannels,hidden_dim_inter,seq_len,padding=padding,act=act)
        self.imp = imputer(nchannels,hidden_dim_imp,act=act)
        self.verbose = verbose
        
    def forward(self,x,m,d):
        '''x = measurements, m = mask, d = time delta between measurements'''
        out = self.inter.forward(x,m,d)
        out = self.imp.forward(x,out,m)
        return out
    
    def fit(self,epochs,optimizer,loss_func,batch_size,x,m,d):
        loss_hist = []
        # Make initial interpolation
        x = lin_interpolation(x)

        # Iterate over epochs
        pbar = tqdm(range(epochs))
        for i in pbar:   
            # shuffle dataset
            indices = torch.randperm(x.shape[0])
            x = x[indices]
            m = m[indices]
            d = d[indices]
            
            temp_loss_hist=[]
            # Iterate over all batches
            for batch in range(int(x.shape[0] / batch_size)):
                x_b = x[batch*batch_size:(batch+1)*batch_size]
                m_b = m[batch*batch_size:(batch+1)*batch_size]
                d_b = d[batch*batch_size:(batch+1)*batch_size]
                # Estimate all values (forward pass)
                output = self.forward(x_b,m_b,d_b)
                # Compute loss
                loss = loss_func(m_b*output,m_b*x_b) # only use loss of actually observed measurements
                # Backward the loss
                optimizer.zero_grad()
                loss.backward()
                temp_loss_hist.append(loss)
                # Update the weights
                optimizer.step()
            if self.verbose and i%10==0:
                # print graph
                x_hat = self.predict(x,m,d)
                missing = (m!=1)
                live_plot(x_hat[0],missing[0],x[0],title=i)
                
            loss_hist.append(torch.stack(temp_loss_hist).mean())
            pbar.set_postfix({'loss': torch.stack(temp_loss_hist).mean()})
             
        return loss_hist
    
    def predict(self,x,m,d,replace=False):
        with torch.no_grad():
            # initial interpolation
            x = lin_interpolation(x)
            # Forward dataset
            out = self.forward(x,m,d)
            if replace:
                observed = (m==1)
                out[observed] = x[observed]
                
        return out        