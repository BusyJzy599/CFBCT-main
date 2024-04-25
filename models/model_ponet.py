import sys
import random
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import torch.nn as nn
import torch
from models.model_utils import *

import numpy as np

class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N

class Omics_Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, attention_dim)
        self.activation = nn.Tanh()

    def forward(self, features):
        features = self.linear(features)
        attention = self.activation(features)
        attention = torch.softmax(attention, dim=0)
        weighted_features = attention * features
        return weighted_features


class PathOmics_Surv(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], model_size_wsi: str='small', 
        model_size_omic: str='small', n_classes=4, dropout=0.25, omic_bag = 'Attention', use_GAP_in_pretrain = False, proj_ratio = 1, image_group_method = 'random'):
        """
        PathOmics model Implementation.
        Inspired by https://github.com/mahmoodlab/MCAT/blob/master/models/model_coattn.py


        Args:
            fusion (str): Late fusion method (Choices: concat, bilinear, or None)
            omic_sizes (List): List of sizes of genomic embeddings
            model_size_wsi (str): Size of WSI encoder (Choices: small or large)
            model_size_omic (str): Size of Genomic encoder (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(PathOmics_Surv, self).__init__()
        
        self.omic_bag = omic_bag
        self.fusion = fusion
        self.use_GAP_in_pretrain = use_GAP_in_pretrain
        self.image_group_method = image_group_method
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [768, 256, 256], "big": [768, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256], 'Methylation':[512, 256],'multi_omics':[512, 256]}
#         self.size_dict_WSI = {"small": [2048, 256, 256], "big": [2048, 512, 384]}
#         self.size_dict_omic = {'small': [256, 256], 'big': [2048, 2048, 2048, 256]}
        
        #self.criterion = SupConLoss(temperature=0.7)
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)

        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        
        ### Projection
        self.path_proj = nn.Linear(size[2], int(size[2] * proj_ratio))#n_classes)
        self.omic_proj = nn.Linear(size[2], int(size[2] * proj_ratio))#n_classes)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        ### PathOmics - WSI
#         self.classifier = Classifier_1fc(mDim, num_cls, droprate)
        self.attention = Attention_Gated(size[2])
        self.dimReduction = DimReduction(size[0], size[2], numLayer_Res=0)
        
        ### PathOmics - Omics
        self.omics_attention_networks = nn.ModuleList([Omics_Attention(omic_sizes[i], size[2]) for i in range(len(omic_sizes))])
        
        self.MHA = nn.MultiheadAttention(size[2], 8)
        self.mutiheadattention_networks = nn.ModuleList([self.MHA for i in range(len(omic_sizes))])
        

    
    def forward(self, x_path, x_omic, x_cluster, mode = ''):
#         print('******')
#         print(x_path.shape)
#         print(x_omic[0].shape)
#         print()
#         ### Bag-Level Representation
#         print("*** 1. Bag-Level Representation (FC Processing) ***")
        
        # image-bag
        
        slide_pseudo_feat = []
        numGroup = len(x_omic)

        
        if self.image_group_method == 'random':
            feat_index = list(range(x_path.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        elif self.image_group_method == 'kmeans_cluster':

            index_chunk_list = [[] for i in range(numGroup)]
            labels = x_cluster
            for index, label in enumerate(labels):
                index_chunk_list[label].append(index)

        for tindex in index_chunk_list:
#             slide_sub_labels.append(tslideLabel)
            subFeat_tensor = torch.index_select(x_path, dim=0, index=torch.LongTensor(tindex).cuda())  # n x 1024
            tmidFeat = self.dimReduction(subFeat_tensor) # n x 256
            tAA = self.attention(tmidFeat).squeeze(0) # n
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x 256
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x 256
            af_inst_feat = tattFeat_tensor # 1 x 256
            slide_pseudo_feat.append(af_inst_feat)
    
        h_path_bag = torch.cat(slide_pseudo_feat, dim=0).unsqueeze(0)  ### numGroup x 256

        
        if self.omic_bag == 'Attention':
            h_omic = [self.omics_attention_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        elif self.omic_bag == 'SNN_Attention':
            h_omic = []
            for idx, sig_feat in enumerate(x_omic):
                snn_feat = self.sig_networks[idx].forward(sig_feat)
                snn_feat = snn_feat.unsqueeze(0).unsqueeze(1)
                attention_feat,_ = self.mutiheadattention_networks[idx].forward(snn_feat,snn_feat,snn_feat)
                h_omic.append(attention_feat.squeeze())
        else:
            h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)### numGroup x 256


        
        h_path_trans = self.path_transformer(h_path_bag)
        h_omic_trans = self.omic_transformer(h_omic_bag)
           
        ### Global Attention Pooling
#         print("*** 4. Global Attention Pooling ***")
        
        if mode == 'pretrain':
            if self.use_GAP_in_pretrain:
                A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
                A_path = torch.transpose(A_path, 1, 0)
                h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
                h_path = self.path_proj(h_path).squeeze()
            else:
                h_path = self.path_proj(h_path_trans).squeeze()
        else:
            A_path, h_path = self.path_attention_head(h_path_trans.squeeze())
            A_path = torch.transpose(A_path, 1, 0)
            h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
            h_path = self.path_rho(h_path).squeeze()
#         print("Final WSI-Level Representation (h^L):\n", h_path.shape)
        

        if mode == 'pretrain':
            if self.use_GAP_in_pretrain:
                A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
                A_omic = torch.transpose(A_omic, 1, 0)
                h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
                h_omic = self.omic_proj(h_omic).squeeze()
            else:
                h_omic = self.omic_proj(h_omic_trans).squeeze()
        else:
            A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze())
            A_omic = torch.transpose(A_omic, 1, 0)
            h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
            h_omic = self.omic_rho(h_omic).squeeze()

#         print("Final Genomic Representation (g^L):\n", h_omic.shape)
#         print()
        
        if mode == 'pretrain':
            return h_path, h_omic #path_embedding, omic_embedding

        else:

            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
            elif self.fusion == 'image':
                h = h_path.squeeze()
            elif self.fusion == 'omics':
                h = h_omic.squeeze()

            ### Survival Layer
            logits = self.classifier(h).unsqueeze(0)
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)

            attention_scores = {}#{'path': A_path, 'omic': A_omic}
            
            return {
                "hazards":hazards,
                "S":S,
                "Y_hat":Y_hat,
            }
#             attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
            


# from MCAT_models.model_utils
def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    
