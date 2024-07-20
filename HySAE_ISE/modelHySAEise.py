# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss

class HySAE(BaseClass):

    def __init__(self, n_ent, n_rel, input_drop, hidden_drop, feature_drop, pad_mode, emb_dim, emb_dim1, max_arity, device,
                 k_size, dil_size):
        super(HySAE, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        self.max_arity = max_arity
        
        self.input_drop = nn.Dropout(input_drop) # input_drop 0.2
        self.hidden_drop = nn.Dropout(hidden_drop) # hidden_drop 0.2
        self.feature_drop = nn.Dropout(feature_drop) # feature_drop 0.3      
 
        ## Controls the linear output of conv_size.
        self.k_size = k_size
        self.dil_size = dil_size
        self.kd_size = self.dil_size*(self.k_size - 1)

        self.pad_size = self.kd_size // 2

        if self.k_size % 2 == 1:
            self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 4
        elif self.k_size % 2 == 0 and self.dil_size % 2 == 0:
            self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 4
        else:
            self.conv_size = ((self.emb_dim1 - 1) * (self.emb_dim2 - 1)) * 8 // 4

        self.pad_mode = pad_mode
        

        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)


        self.conv_layer_2a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 2), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))
        
        self.conv_layer_3a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 3), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))
        
        self.conv_layer_4a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 4), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))
        
        self.conv_layer_5a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 5), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))
        
        self.conv_layer_6a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 6), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))
        
        self.conv_layer_7a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 7), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))
        
        self.conv_layer_8a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 8), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))
       
        self.conv_layer_9a = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.k_size, self.k_size, 9), 
                                       padding=(self.pad_size,self.pad_size,0), padding_mode = self.pad_mode, dilation=(self.dil_size,self.dil_size,1))

    
    
        self.pool = torch.nn.MaxPool3d((4, 1, 1))
        
        self.fc_layer = nn.Linear(in_features=self.conv_size, out_features=self.emb_dim)
        
        
        

        self.bn1 = nn.BatchNorm3d(num_features=1)
        # self.bn2 = nn.BatchNorm3d(num_features=4)
        # self.bn3 = nn.BatchNorm2d(num_features=32)
        # self.bn4 = nn.BatchNorm1d(num_features=self.conv_size)
        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))



        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)
        
        nn.init.xavier_uniform_(self.conv_layer_2a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_3a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_4a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_5a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_6a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_7a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_8a.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_9a.weight.data)
        
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        
    


    def conv3d_dilated(self, concat_input):
        r = concat_input[:, 0, :].view(-1, 1, self.emb_dim1, self.emb_dim2)

        if concat_input.shape[1] == 2:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            
            cube = torch.cat((r, e1), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_2a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 3:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
                                   
            cube = torch.cat((r, e1, e2), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_3a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 4:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            
            cube = torch.cat((r, e1, e2, e3), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_4a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 5:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            
            cube = torch.cat((r, e1, e2, e3, e4), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_5a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 6:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            
            cube = torch.cat((r, e1, e2, e3, e4, e5), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_6a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 7:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_7a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 8:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_8a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if concat_input.shape[1] == 9:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e8 = concat_input[:, 8, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7, e8), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_9a(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        x = x.view(-1, self.conv_size)
        x = self.feature_drop(x) # feature_drop
        
        return x



    def forward(self, rel_idx, ent_idx, miss_ent_domain):

        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = self.ent_embeddings[ent_idx]
        
        pos = [i for i in range(ent_idx.shape[1]+1) if i + 1 != miss_ent_domain]
        pos = torch.tensor(pos).to(self.device)
        pos = pos.unsqueeze(0).expand_as(ent_idx)
        ents = ents + self.pos_embeddings(pos)
        
        concat_input = torch.cat((r, ents), dim=1)   
        concat_input = self.input_drop(concat_input) # input_drop
        
        v1 = self.conv3d_dilated(concat_input)
        
        v_out = v1
        x = self.hidden_drop(v_out) # hidden_drop 
        x = self.fc_layer(x)

        

        miss_ent_domain = torch.LongTensor([miss_ent_domain-1]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos
        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)

        return scores