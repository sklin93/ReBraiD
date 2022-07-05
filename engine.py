import torch.optim as optim
import torch.nn as nn
from model import *
from Utils.util import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform(m.weight)
        # torch.nn.init.xavier_uniform_(m.weight)
        # m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Sequential or type(m) == nn.ModuleList:
        for k in m:
            init_weights(k)

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid,
                 dropout, lrate, wdecay, device, supports, gcn_bool,
                 addaptadj, aptinit, kernel_size, total_layers=0, layers=0,
                 batch_size=None, no_SC=False, noT=False, pool=True,
                 use_corrcoef=False, out_class=6, verbose=True):

        supports_len = 0
        for k in supports: 
            supports_len = len(supports[k])
            break
        if gcn_bool:
            if use_corrcoef:
                supports_len += 1
            if addaptadj:
                supports_len += 1

        # self.model = mls_gat_classifier(device, num_nodes, dropout,
        # self.model = mls_coarsened_classifier(device, num_nodes, dropout,
        self.model = mls_classifier(device, num_nodes, dropout,
                                    supports_len, batch_size, gcn_bool=gcn_bool,
                                    addaptadj=addaptadj, in_dim=in_dim,
                                    seq_len=seq_length, residual_channels=nhid,
                                    dilation_channels=nhid,
                                    skip_channels=nhid*8, end_channels=nhid*16,
                                    kernel_size=kernel_size,
                                    total_layers=total_layers,
                                    block_layers=layers, noT=noT, pool=pool,
                                    use_corrcoef=use_corrcoef,
                                    out_class=out_class)

        # self.model.apply(init_weights)
        self.model.to(device)
        if verbose:
            print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate,
                                    weight_decay=wdecay)
        
        self.scaler = scaler
        self.clip = 5
        self.supports = supports
        self.aptinit = aptinit
        self.state = None
        self.device = device
        self.lrate = lrate
        self.wdecay = wdecay
        self.no_SC = no_SC
        self.out_class = out_class

    def set_state(self, state):
        assert state in ['train', 'val', 'test']
        self.state = state
        self.state_supports = [torch.tensor(i).to(
                                    self.device) for i in self.supports[state]]

    def train_classifier(self, input, real_F, real_E, assign_dict, adj_idx,
                         lr_update=None):
        self.model.train()
        if lr_update:
            self.lrate *= lr_update
            print('Updating lr to', self.lrate)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lrate,
                                        weight_decay=self.wdecay)
        self.optimizer.zero_grad()
        
        assert self.state is not None, 'set train/val/test state first'

        supports = self.state_supports
        supports = [s[adj_idx] for s in supports]
        
        aptinit = self.aptinit
        if aptinit is not None:
            aptinit = aptinit[self.state]
            if aptinit is not None:
                aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

        if self.no_SC:
            F = self.model(input)
        else:
            F = self.model(input, torch.stack(supports, 1), aptinit)

        ##### loss #####
        real_F = real_F.to(self.device).squeeze()
        if self.out_class == 1:
            loss = nn.BCEWithLogitsLoss()(F, real_F.type_as(F))
        else:
            loss = nn.CrossEntropyLoss(weight=None)(F, real_F)

        loss.mean().backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        if self.out_class == 1:
            _F = torch.sigmoid(F)
            _F[_F > 0.5] = 1
            _F[_F != 1] = 0
            _F =  _F.detach().cpu().numpy()
            _real_F = real_F.detach().cpu().numpy()
            accuracy = round(accuracy_score(_real_F, _F) * 100, 2)
            f1 =  round(f1_score(_real_F, _F, average='weighted') * 100, 2)
        else:
            _real_F = real_F.detach().cpu().numpy()
            _F =  F.detach().cpu().numpy()
            accuracy = round(accuracy_score(_real_F, np.argmax(_F, 1)) * 100, 2)
            f1 =  round(f1_score(_real_F, np.argmax(_F, 1),
                                 average='weighted') * 100, 2)
                          
        return loss.item(), accuracy, f1

    def eval_classifier(self, input, real_F, real_E, assign_dict, adj_idx,
                        viz=False):
        self.model.eval()

        supports = self.state_supports
        supports = [s[adj_idx] for s in supports]

        aptinit = self.aptinit
        if aptinit is not None:
            aptinit = aptinit[self.state]
            if aptinit is not None:
                aptinit = torch.Tensor(aptinit[adj_idx]).to(self.device)

        with torch.no_grad():
            if self.no_SC:
                F = self.model(input, viz=viz)
            else:
                F = self.model(input, torch.stack(supports, 1), aptinit,
                               viz=viz)
        ##### loss #####
        real_F = real_F.to(self.device).squeeze()
        if self.out_class == 1:
            loss = nn.BCEWithLogitsLoss()(F, real_F.type_as(F))
        else:
            loss = nn.CrossEntropyLoss(weight=None)(F, real_F)

        ##### metrics #####
        if self.out_class == 1:
            _F = torch.sigmoid(F)
            _F[_F > 0.5] = 1
            _F[_F != 1] = 0
            _F =  _F.cpu().numpy()
            _real_F = real_F.cpu().numpy()

            accuracy = round(accuracy_score(_real_F, _F) * 100, 2)
            f1 =  round(f1_score(_real_F, _F, average='weighted') * 100, 2)
            cm = confusion_matrix(_real_F, F)
        else:
            _real_F = real_F.cpu().numpy()
            _F =  F.cpu().numpy()
            accuracy = round(accuracy_score(_real_F, np.argmax(_F, 1)) * 100, 2)
            f1 =  round(f1_score(_real_F, np.argmax(_F, 1),
                average='weighted') * 100, 2)
            cm = confusion_matrix(_real_F, np.argmax(_F, 1), labels=np.arange(
                self.out_class))

        return loss.item(), accuracy, f1, cm
