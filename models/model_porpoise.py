import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *



class PorpoiseAMIL(nn.Module):
    def __init__(self, size_arg = "small", n_classes=4):
        super(PorpoiseAMIL, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        self.classifier = nn.Linear(size[1], n_classes)
        initialize_weights(self)
                
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        h = kwargs['x_path']

        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0)

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A

        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        h  = self.classifier(M)
        return h

    def get_slide_features(self, **kwargs):
        h = kwargs['x_path']

        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0)

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A

        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        return M


### MMF (in the PORPOISE Paper)
class PorpoiseMMF(nn.Module):
    def __init__(self, 
        omic_input_dim,
        path_input_dim=768, 
        fusion='bilinear', 
        dropout=0.25,
        n_classes=4, 
        scale_dim1=8, 
        scale_dim2=8, 
        gate_path=1, 
        gate_omic=1, 
        skip=True, 
        dropinput=0.10,
        use_mlp=False,
        size_arg = "small",
        ):
        super(PorpoiseMMF, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [path_input_dim, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.n_classes = n_classes

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        if dropinput:
            fc = [nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        else:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Constructing Genomic SNN
        if self.fusion is not None:
            if use_mlp:
                Block = MLP_Block
            else:
                Block = SNN_Block

            hidden = self.size_dict_omic['small']
            fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        
            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=256)
            else:
                self.mm = None

        self.classifier_mm = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier_mm = self.classifier_mm.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path)

        x_omic = torch.cat([kwargs['x_omic%d' % i] for i in range(1,7)]).view(1,-1)
        h_omic = self.fc_omic(x_omic)
        if self.fusion == 'bilinear':
            h_mm = self.mm(h_path, h_omic)
        elif self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic], axis=1))
        elif self.fusion == 'lrb':
            h_mm  = self.mm(h_path, h_omic) # logits needs to be a [1 x 4] vector 
            return h_mm

        h_mm  = self.classifier_mm(h_mm) # logits needs to be a [B x 4] vector      
        assert len(h_mm.shape) == 2 and h_mm.shape[1] == self.n_classes

        logits = h_mm
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return {
            "hazards":hazards,
            "S":S,
            "Y_hat":Y_hat,
        }

    def captum(self, h, X):
        A, h = self.attention_net(h)  
        A = A.squeeze(dim=2)

        A = F.softmax(A, dim=1) 
        M = torch.bmm(A.unsqueeze(dim=1), h).squeeze(dim=1) #M = torch.mm(A, h)
        M = self.rho(M)
        O = self.fc_omic(X)

        if self.fusion == 'bilinear':
            MM = self.mm(M, O)
        elif self.fusion == 'concat':
            MM = self.mm(torch.cat([M, O], axis=1))
            
        logits  = self.classifier(MM)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1)
        return risk