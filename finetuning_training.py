import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import os
from datetime import datetime
from matplotlib import pyplot as plt

from dataset.dataset_class import FineTuningImagesDataset
from network.model import Generator, Discriminator
from loss.loss_discriminator import LossDSCfake, LossDSCreal
from loss.loss_generator import LossGF

from params.params import (
    path_to_chkpt, path_to_Wi, finetuning_batch_size as batch_size, frame_shape,
    path_to_identity_img, path_to_identity_seg, path_to_identity_embedding,
    path_to_finetuned_model, VGG19_weight_path, VGGFace_body_path,
    VGGFace_weight_path)

"""Hyperparameters and config"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_embedding = path_to_identity_embedding
path_to_save = path_to_finetuned_model

path_to_images = path_to_identity_img
path_to_segs = path_to_identity_seg

"""Create dataset and net"""
dataset = FineTuningImagesDataset(path_to_images, path_to_segs, device)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

ei_hat = torch.load(path_to_embedding, map_location=cpu)
ei_hat = ei_hat['ei'].unsqueeze(0).to(device)#unsqueeze(-1)

G = Generator(frame_shape, finetuning=True)
D = Discriminator(batch_size, dataset.__len__(), path_to_Wi, finetuning=True)

G.train()
D.train()

Ep = models.mobilenet_v2(num_classes=256).to(device)
Ep.eval()

optimizer = optim.Adam(
        params=list(G.parameters()) + list(D.parameters()),
        lr=2e-4,
        amsgrad=False)

"""Criterion"""
criterionG = LossGF(
        VGG19_weight_path=VGG19_weight_path,
        VGGFace_body_path=VGGFace_body_path,
        VGGFace_weight_path=VGGFace_weight_path,
        device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 100

#Warning if checkpoint inexistant
if not os.path.isfile(path_to_chkpt):
    print('ERROR: cannot find checkpoint')
if os.path.isfile(path_to_save):
    path_to_chkpt = path_to_save

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
checkpoint['D_state_dict']['W_i'] = torch.randn(batch_size,768,1)#change W_i for finetuning
checkpoint['D_state_dict']['w_prime'] = torch.randn(batch_size,768,1)#change W_i for finetuning

G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'], strict = False)
Ep.load_state_dict(checkpoint['Ep_state_dict'])

G.to(device)
D.to(device)
Ep.to(device)

"""Training"""
batch_start = datetime.now()

os.makedirs('vis', exist_ok=True)
cont = True
while cont:
    for epoch in range(num_epochs):
        for i_batch, (pose_aug, pose_img, pose_seg) in enumerate(dataLoader):
            pose_aug = pose_aug.to(device)
            pose_img = pose_img.to(device)
            pose_seg = pose_seg.to(device)
            
            with torch.no_grad():
                ep_hat = Ep(pose_aug)
                e_hat = torch.cat([ei_hat.expand(ep_hat.shape[0], 512), ep_hat], dim=1).unsqueeze(-1) #B,768
            
            with torch.autograd.enable_grad():
                #zero the parameter gradients
                optimizer.zero_grad()
    
                #forward
                #train G and D
                is_hat = G(e_hat)
                i_hat = is_hat[:,:3]
                s_hat = is_hat[:,3,None]
                
                x = torch.mul(pose_img, pose_seg)
                x_hat = torch.mul(i_hat, s_hat)
                r_hat, D_hat_res_list = D(x_hat, e_finetuning=e_hat, i=0)
                r, D_res_list = D(x, e_finetuning=e_hat, i=0)
    
                lossG = criterionG(
                        x=x,
                        x_hat=x_hat,
                        s=pose_seg,
                        s_hat=s_hat,
                        r_hat=r_hat,
                        D_res_list=D_res_list,
                        D_hat_res_list=D_hat_res_list)
    
                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)
    
                lossD = lossDreal + lossDfake
                loss = lossG.mean() + lossD
            
                loss.backward(retain_graph=False)
                optimizer.step()
    
        batch_end = datetime.now()
        avg_time = (batch_end - batch_start) / 10
        print('\n\navg batch time for batch size of', x.shape[0],':',avg_time)
        
        batch_start = datetime.now()
        
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
              % (epoch, num_epochs, i_batch, len(dataLoader),
                 lossD.item(), lossG.item(), r.mean(), r_hat.mean()))
        
        outx = torch.cat([i.permute(1,2,0) for i in x], dim=1) * 255
        outxaug = torch.cat([i.permute(1,2,0) for i in pose_aug], dim=1) * 255
        outxhat = torch.cat([i.permute(1,2,0) for i in x_hat], dim=1) * 255
        out = torch.cat([outx, outxaug, outxhat], dim=0)
        out = out.type(torch.uint8).to(cpu).numpy()
        plt.imsave("vis/{:03d}_XXXXX.png".format(epoch), out)
    
    num_epochs = int(input('Num epoch further?\n'))
    cont = num_epochs != 0

print('Saving finetuned model...')
torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'Ep_state_dict': Ep.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path_to_save)
print('...Done saving latest')
