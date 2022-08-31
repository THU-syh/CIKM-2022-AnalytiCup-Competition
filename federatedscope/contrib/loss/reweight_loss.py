from federatedscope.register import register_criterion


import torch
import torch.nn as nn
import torch.nn.functional as F

class ReweightLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.Tensor([1.0,1.0])
        self.criterion = nn.NLLLoss()

    def set_weight(self,weight):
        self.weight = weight

    def forward(self,pred,target):
        weight = self.weight.cuda()/0.5

        pred = pred.view(-1,1)
        pred = torch.cat((1-pred,pred),dim=1) # Batch_size * 2
        log_p = torch.log(pred+1e-4)/weight

        target = target.squeeze(-1).long()
        loss = self.criterion(log_p,target)
        return loss

def create_reweight_loss(type, device):
    if type == "ReweightLoss":
        criterion = ReweightLoss().to(device)
        return criterion


register_criterion("ReweightLoss", create_reweight_loss)