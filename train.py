"""Main"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.ion()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from dataset.dataset_class import PreprocessDataset
# from dataset.video_extraction_conversion import *
from loss.loss_discriminator import LossDSCfake, LossDSCreal
from loss.loss_generator import LossG
# from network.blocks import *
from network.model import Generator, Discriminator
from tqdm import tqdm

from params.params import K, path_to_chkpt, path_to_backup, batch_size, path_to_images, path_to_segs, frame_shape, path_to_Wi

"""Create dataset and net"""
display_training = False
device = torch.device("cuda:0")
cpu = torch.device("cpu")
dataset = PreprocessDataset(K=K, path_to_images=path_to_images, path_to_segs=path_to_segs, path_to_Wi=path_to_Wi)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=1,
                        pin_memory=True,
                        drop_last = True)

G = nn.DataParallel(Generator(frame_shape).to(device))
Ei = nn.DataParallel(models.resnext50_32x4d(num_classes=512).to(device))
Ep = nn.DataParallel(models.mobilenet_v2(num_classes=256).to(device))
D = nn.DataParallel(Discriminator(dataset.__len__(), path_to_Wi).to(device))

G.train()
Ei.train()
Ep.train()
D.train()


optimizerG = optim.Adam(params = list(Ei.parameters()) + list(Ep.parameters()) + list(G.parameters()),
                        lr=5e-5,
                        amsgrad=False)
optimizerD = optim.Adam(params = D.parameters(),
                        lr=2e-4,
                        amsgrad=False)

"""Criterion"""
criterionG = LossG(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
                   VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
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
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
            }, path_to_chkpt)
    print('...Done')


"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
Ei.module.load_state_dict(checkpoint['Ei_state_dict'])
Ep.module.load_state_dict(checkpoint['Ep_state_dict'])
G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
D.module.load_state_dict(checkpoint['D_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] +1
optimizerG.load_state_dict(checkpoint['optimizerG'])
optimizerD.load_state_dict(checkpoint['optimizerD'])

G.train()
Ei.train()
Ep.train()
D.train()

"""Training"""
batch_start = datetime.now()
pbar = tqdm(dataLoader, leave=True, initial=0)
if not display_training:
    matplotlib.use('agg')

#3200000: total iterations
for epoch in range(epochCurrent, num_epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)
    pbar.set_postfix(epoch=epoch)
    for i_batch, (identity_imgs, pose_img, pose_aug, pose_seg, idxs, W_i) in enumerate(pbar, start=0):
        
        identity_imgs = identity_imgs.to(device)
        pose_img = pose_img.to(device)
        pose_seg = pose_seg.to(device)
        W_i = W_i.squeeze(-1).transpose(0,1).to(device).requires_grad_()
        
        D.module.load_W_i(W_i)
        
        if i_batch % 1 == 0:
            with torch.autograd.enable_grad():
                #zero the parameter gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()

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
                r_hat, D_hat_res_list = D(x_hat, idxs)
                with torch.no_grad():
                    r, D_res_list = D(x, idxs)
                """####################################################################################################################################################
                r, D_res_list = D(x, g_y, i)"""

                lossG = criterionG(
                    x=x,
                    x_hat=x_hat,
                    s=pose_seg,
                    s_hat=s_hat,
                    r_hat=r_hat,
                    D_res_list=D_res_list,
                    D_hat_res_list=D_hat_res_list,
                    e_vectors=ei_vectors,
                    W=D.module.W_i,
                    i=idxs)
                """####################################################################################################################################################
                lossD = criterionDfake(r_hat) + criterionDreal(r)
                loss = lossG + lossD
                loss.backward(retain_graph=False)
                optimizerG.step()
                optimizerD.step()"""
                
                lossG.backward(retain_graph=False)
                optimizerG.step()
                #optimizerD.step()
            
            with torch.autograd.enable_grad():
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                x_hat.detach_().requires_grad_()
                r_hat, D_hat_res_list = D(x_hat, idxs)
                lossDfake = criterionDfake(r_hat)

                r, D_res_list = D(x, idxs)
                lossDreal = criterionDreal(r)
                
                lossD = lossDfake + lossDreal
                lossD.backward(retain_graph=False)
                optimizerD.step()
                
                optimizerD.zero_grad()
                r_hat, D_hat_res_list = D(x_hat, idxs)
                lossDfake = criterionDfake(r_hat)

                r, D_res_list = D(x, idxs)
                lossDreal = criterionDreal(r)
                
                lossD = lossDfake + lossDreal
                lossD.backward(retain_graph=False)
                optimizerD.step()

        for enum, idx in enumerate(idxs):
            wi_path = path_to_Wi+'/W_'+str(idx.item()//256)+'/W_'+str(idx.item())+'.tar'
            torch.save({'W_i': D.module.W_i[:,enum].unsqueeze(-1)}, wi_path)
                    

        # Output training stats
        if i_batch % 1 == 0 and i_batch > 0:
            #batch_end = datetime.now()
            #avg_time = (batch_end - batch_start) / 100
            # print('\n\navg batch time for batch size of', x.shape[0],':',avg_time)
            
            #batch_start = datetime.now()
            
            # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
            #       % (epoch, num_epochs, i_batch, len(dataLoader),
            #          lossD.item(), lossG.item(), r.mean(), r_hat.mean()))
            pbar.set_postfix(epoch=epoch, r=r.mean().item(), rhat=r_hat.mean().item(), lossG=lossG.item())

            if display_training:
                plt.figure(figsize=(10,10))
                plt.clf()
                out = (x_hat[0]*255).permute(1,2,0)
                for img_no in range(1,x_hat.shape[0]//16):
                    out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                fig = out

                plt.clf()
                out = (x[0]*255).permute(1,2,0)
                for img_no in range(1,x.shape[0]//16):
                    out = torch.cat((out, (x[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                fig = np.concatenate((fig, out), 0)

                plt.imshow(fig)
                plt.xticks([])
                plt.yticks([])
                plt.draw()
                plt.pause(0.001)
            
            

        if i_batch % 1000 == 999:
            lossesD.append(lossD.item())
            lossesG.append(lossG.item())

            if display_training:
                plt.clf()
                plt.plot(lossesG) #blue
                plt.plot(lossesD) #orange
                plt.show()

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
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                    }, path_to_chkpt)
            out = (x_hat[0]*255).permute(1,2,0)
            for img_no in range(1,2):
                out = torch.cat((out, (x_hat[img_no]*255).permute(1,2,0)), dim = 1)
            out = out.type(torch.uint8).to(cpu).numpy()
            plt.imsave("vis/{:03d}_{:05d}.png".format(epoch, i_batch), out)
            print('...Done saving latest')
            
    if epoch%1 == 0:
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
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
                }, path_to_backup)
        out = (x_hat[0]*255).permute(1,2,0)
        for img_no in range(1,2):
            out = torch.cat((out, (x_hat[img_no]*255).permute(1,2,0)), dim = 1)
        out = out.type(torch.uint8).to(cpu).numpy()
        plt.imsave("vis/{:03d}_XXXXX.png".format(epoch,), out)
        print('...Done saving latest')
