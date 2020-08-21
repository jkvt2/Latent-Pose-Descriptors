import torch
import torch.nn as nn
import imp
import torchvision
from network.model import Cropped_VGG19


class LossCnt(nn.Module):
    def __init__(self, VGG19_body_path, VGG19_weight_path, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossCnt, self).__init__()
        
        MainModel = imp.load_source('MainModel', VGG19_body_path)
        full_VGG19 = torch.load(VGG19_weight_path, map_location = 'cpu')
        cropped_VGG19 = Cropped_VGG19()
        cropped_VGG19.load_state_dict(full_VGG19.state_dict(), strict = False)
        self.VGG19 = cropped_VGG19
        self.VGG19.eval()
        self.VGG19.to(device)
        
        MainModel = imp.load_source('MainModel', VGGFace_body_path)
        full_VGGFace = torch.load(VGGFace_weight_path, map_location = 'cpu')
        cropped_VGGFace = Cropped_VGG19()
        cropped_VGGFace.load_state_dict(full_VGGFace.state_dict(), strict = False)
        self.VGGFace = cropped_VGGFace
        self.VGGFace.eval()
        self.VGGFace.to(device)

        self.l1_loss = nn.L1Loss()

    def forward(self, x, x_hat, vgg19_weight=1.5e-1, vggface_weight=2.5e-2):        
        """Retrieve vggface feature maps"""
        with torch.no_grad(): #no need for gradient compute
            vgg_x_features = self.VGGFace(x) #returns a list of feature maps at desired layers

        with torch.autograd.enable_grad():
            vgg_xhat_features = self.VGGFace(x_hat)

        lossface = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            lossface += self.l1_loss(x_feat, xhat_feat)

        """Retrieve vgg19 feature maps"""
        with torch.no_grad(): #no need for gradient compute
            vgg_x_features = self.VGG19(x) #returns a list of feature maps at desired layers

        with torch.autograd.enable_grad():
            vgg_xhat_features = self.VGG19(x_hat)

        loss19 = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            loss19 += self.l1_loss(x_feat, xhat_feat)

        loss = vgg19_weight * loss19 + vggface_weight * lossface

        return loss


class LossAdv(nn.Module):
    def __init__(self, FM_weight=1e1):
        super(LossAdv, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.FM_weight = FM_weight
        
    def forward(self, r_hat, D_res_list, D_hat_res_list):
        lossFM = 0
        for res, res_hat in zip(D_res_list, D_hat_res_list):
            lossFM += self.l1_loss(res, res_hat)
        
        return -r_hat.mean() + lossFM * self.FM_weight


class LossMatch(nn.Module):
    def __init__(self, device, match_weight=1e1):
        super(LossMatch, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.match_weight = match_weight
        self.device = device
        
    def forward(self, e_vectors, W, i):
        return self.l1_loss(e_vectors, W) * self.match_weight


class LossDice(nn.Module):
    def __init__(self,):
        super(LossDice, self).__init__()
        
    def forward(self, s, s_hat, smooth=1.0):
    
        iflat = s.contiguous().view(-1)
        tflat = s_hat.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))

    
class LossG(nn.Module):
    """
    Loss for generator meta training
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, VGG19_body_path, VGG19_weight_path, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossG, self).__init__()
        
        self.lossCnt = LossCnt(VGG19_body_path, VGG19_weight_path, VGGFace_body_path, VGGFace_weight_path, device)
        self.lossAdv = LossAdv()
        self.lossMatch = LossMatch(device=device)
        self.lossDice = LossDice()
        
    def forward(self, x, x_hat, s, s_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, W, i):
        loss_cnt = self.lossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        loss_match = self.lossMatch(e_vectors, W, i)
        loss_dice = self.lossDice(s_hat, s)
        #print(loss_cnt.item(), loss_adv.item(), loss_match.item())
        return loss_cnt + loss_adv + loss_match + loss_dice

class LossGF(nn.Module):
    """
    Loss for generator finetuning
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, VGG19_body_path, VGG19_weight_path, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossGF, self).__init__()
        
        self.lossCnt = LossCnt(VGG19_body_path, VGG19_weight_path, VGGFace_body_path, VGGFace_weight_path, device)
        self.lossAdv = LossAdv()
        self.lossDice = LossDice()
        
    def forward(self, x, x_hat, s, s_hat, r_hat, D_res_list, D_hat_res_list):
        loss_cnt = self.LossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        loss_dice = self.lossDice(s_hat, s)
        return loss_cnt + loss_adv + loss_dice
