"""Main"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from datetime import datetime

from matplotlib import pyplot as plt
import os

from dataset.dataset_class import PreprocessDataset
from loss.loss_discriminator import LossDSCfake, LossDSCreal
from loss.loss_generator import LossG
from network.model import Generator, Discriminator
from tqdm import tqdm

from params.params import (
    K, path_to_chkpt, path_to_backup, batch_size, path_to_images, path_to_segs,
    frame_shape, path_to_Wi, VGG19_weight_path, VGGFace_body_path, VGGFace_weight_path)

"""Create dataset and net"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
dataset = PreprocessDataset(K=K, path_to_images=path_to_images, path_to_segs=path_to_segs, path_to_Wi=path_to_Wi)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=1,
                        pin_memory=True,
                        drop_last = True)

G = nn.DataParallel(Generator(frame_shape).to(device))
if hasattr(models, 'resnext50_32x4d'):
    Ei = nn.DataParallel(models.resnext50_32x4d(num_classes=512).to(device))
    Ep = nn.DataParallel(models.mobilenet_v2(num_classes=256).to(device))
else:
    from network.resnet import resnext50_32x4d
    from network.mobilenet import mobilenet_v2
    Ei = nn.DataParallel(resnext50_32x4d(num_classes=512).to(device))
    Ep = nn.DataParallel(mobilenet_v2(num_classes=256).to(device))
D = nn.DataParallel(Discriminator(batch_size, dataset.__len__(), path_to_Wi).to(device))

G.train()
Ei.train()
Ep.train()
D.train()

optimizer = optim.Adam(
        params=list(Ei.parameters()) + list(Ep.parameters()) + list(G.parameters()) + list(D.parameters()),
        lr=5e-5,
        amsgrad=False)

"""Criterion"""
criterionG = nn.DataParallel(LossG(
        VGG19_weight_path=VGG19_weight_path,
        VGGFace_body_path=VGGFace_body_path,
        VGGFace_weight_path=VGGFace_weight_path,
        device=device))
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 75*5

#initiate checkpoint if inexistant
if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    G.apply(init_weights)
    D.apply(init_weights)
    Ei.apply(init_weights)
    Ep.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'Ei_state_dict': Ei.module.state_dict(),
            'Ep_state_dict': Ep.module.state_dict(),
            'G_state_dict': G.module.state_dict(),
            'D_state_dict': D.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizer': optimizer.state_dict(),
            }, path_to_chkpt)
    print('...Done')


"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
Ei.module.load_state_dict(checkpoint['Ei_state_dict'])
Ep.module.load_state_dict(checkpoint['Ep_state_dict'])
G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
checkpoint['D_state_dict']['W_i'] = nn.Parameter(torch.randn(batch_size,768,1))
D.module.load_state_dict(checkpoint['D_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] +1
optimizer.load_state_dict(checkpoint['optimizer'])

G.train()
Ei.train()
Ep.train()
D.train()

"""Training"""
batch_start = datetime.now()
pbar = tqdm(dataLoader, leave=True, initial=0)
os.makedirs('vis', exist_ok=True)

for epoch in range(epochCurrent, num_epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)
    pbar.set_postfix(epoch=epoch)
    for i_batch, (identity_imgs, pose_img, pose_aug, pose_seg, idxs, W_i) in enumerate(pbar, start=0):
        
        identity_imgs = identity_imgs.to(device)
        pose_img = pose_img.to(device)
        pose_seg = pose_seg.to(device)
        W_i = W_i.to(device).requires_grad_()
        
        D.module.load_W_i(W_i)
        
        with torch.autograd.enable_grad():
            #zero the parameter gradients
            optimizer.zero_grad()

            #forward
            # Calculate average encoding vector for video
            identity_imgs_rs = identity_imgs.view(-1,
                identity_imgs.shape[-3],
                identity_imgs.shape[-2],
                identity_imgs.shape[-1]) #BxK,3,256,256

            ei_vectors = Ei(identity_imgs_rs) #BxK,512
            ei_vectors = ei_vectors.view(-1, identity_imgs.shape[1], 512) #B,K,512
            ei_hat = ei_vectors.mean(dim=1)
            
            ep_hat = Ep(pose_aug) #B,256
            
            e_hat = torch.cat([ei_hat, ep_hat], dim=1).unsqueeze(-1) #B,768

            #train G and D
            is_hat = G(e_hat)
            i_hat = is_hat[:,:3]
            s_hat = is_hat[:,3,None]
            x = torch.mul(pose_img, pose_seg)
            x_hat = torch.mul(i_hat, s_hat)
            r_hat, D_hat_res_list = D(x_hat)
            r, D_res_list = D(x)

            lossG = criterionG(
                x=x,
                x_hat=x_hat,
                s=pose_seg,
                s_hat=s_hat,
                r_hat=r_hat,
                D_res_list=D_res_list,
                D_hat_res_list=D_hat_res_list,
                e_vectors=e_hat,
                W=D.module.W_i)
            
            lossDfake = criterionDfake(r_hat)
            lossDreal = criterionDreal(r)
            
            lossD = lossDfake + lossDreal
            loss = lossG.mean() + 8 * lossD
            
            loss.backward(retain_graph=False)
            optimizer.step()

        for enum, idx in enumerate(idxs):
            wi_path = path_to_Wi+'/W_'+str(idx.item()//256)+'/W_'+str(idx.item())+'.tar'
            torch.save({'W_i': D.module.W_i[enum].detach().cpu()}, wi_path)
                    
        if i_batch % 1000 == 999:
            lossesD.append(lossD.mean().item())
            lossesG.append(lossG.mean().item())

            print('Saving latest...')
            torch.save({
                    'epoch': epoch,
                    'lossesG': lossesG,
                    'lossesD': lossesD,
                    'Ei_state_dict': Ei.module.state_dict(),
                    'Ep_state_dict': Ep.module.state_dict(),
                    'G_state_dict': G.module.state_dict(),
                    'D_state_dict': D.module.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': i_batch,
                    'optimizer': optimizer.state_dict(),
                    }, path_to_chkpt)
            outx = torch.cat([i.permute(1,2,0) for i in x], dim=1) * 255
            outxhat = torch.cat([i.permute(1,2,0) for i in x_hat], dim=1) * 255
            out = torch.cat([outx, outxhat], dim=0)
            out = out.type(torch.uint8).to(cpu).numpy()
            plt.imsave("vis/{:03d}_{:05d}.png".format(epoch, i_batch), out)
            print('...Done saving latest')
            
    print('Saving latest...')
    torch.save({
            'epoch': epoch+1,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'Ei_state_dict': Ei.module.state_dict(),
            'Ep_state_dict': Ep.module.state_dict(),
            'G_state_dict': G.module.state_dict(),
            'D_state_dict': D.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizer': optimizer.state_dict(),
            }, path_to_backup)
    outx = torch.cat([i.permute(1,2,0) for i in x], dim=1) * 255
    outxhat = torch.cat([i.permute(1,2,0) for i in x_hat], dim=1) * 255
    out = torch.cat([outx, outxhat], dim=0)
    out = out.type(torch.uint8).to(cpu).numpy()
    plt.imsave("vis/{:03d}_XXXXX.png".format(epoch,), out)
    print('...Done saving latest')
