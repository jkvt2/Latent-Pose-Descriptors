import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import os
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib

# import numpy as np

from dataset.dataset_class import FineTuningImagesDataset
from network.model import Generator, Discriminator
from loss.loss_discriminator import LossDSCfake, LossDSCreal
from loss.loss_generator import LossGF

from params.params import path_to_chkpt, path_to_Wi, batch_size, frame_shape

"""Hyperparameters and config"""
display_training = True
if not display_training:
	matplotlib.use('agg')
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_embedding = 'e_hat_images.tar'
path_to_save = 'finetuned_model.tar'
# path_to_video = 'examples/fine_tuning/test_video.mp4'
path_to_images = '/media/vince/storage/dl/data/identity_source/img'
path_to_segs = '/media/vince/storage/dl/data/identity_source/seg'

"""Create dataset and net"""
# choice = ''
# while choice != '0' and choice != '1':
#     choice = input('What source to finetune on?\n0: Video\n1: Images\n\nEnter number\n>>')
# if choice == '0': #video
#     dataset = FineTuningVideoDataset(path_to_video, device)
# else: #Images
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

optimizerG = optim.Adam(params = G.parameters(), lr=5e-5)
optimizerD = optim.Adam(params = D.parameters(), lr=2e-4)


"""Criterion"""
criterionG = LossGF(
        VGG19_weight_path='vgg19-dcbb9e9d.pth',
        VGGFace_body_path='Pytorch_VGGFACE_IR.py',
        VGGFace_weight_path='Pytorch_VGGFACE.pth',
        device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 1000

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

# """Change to finetuning mode"""
# G.finetuning_init()
# D.finetuning_init()

G.to(device)
D.to(device)
Ep.to(device)

"""Training"""
batch_start = datetime.now()

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
                optimizerG.zero_grad()
                optimizerD.zero_grad()
    
                #forward
                #train G and D
                is_hat = G(e_hat)
                i_hat = is_hat[:,:3]
                s_hat = is_hat[:,3,None]
                
                x = torch.mul(pose_img, pose_seg)
                x_hat = torch.mul(i_hat, s_hat)
                r_hat, D_hat_res_list = D(x_hat, e_finetuning=e_hat, i=0)
                with torch.no_grad():
                    r, D_res_list = D(x, e_finetuning=e_hat, i=0)
    
                lossG = criterionG(
                        x=x,
                        x_hat=x_hat,
                        s=pose_seg,
                        s_hat=s_hat,
                        r_hat=r_hat,
                        D_res_list=D_res_list,
                        D_hat_res_list=D_hat_res_list)
    
                lossG.backward(retain_graph=False)
                optimizerG.step()
                
                
                #train D
                optimizerD.zero_grad()
                x_hat.detach_().requires_grad_()
                r_hat, D_hat_res_list = D(x_hat, e_finetuning=e_hat, i=0)
                r, D_res_list = D(x, e_finetuning=e_hat, i=0)
    
                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)
    
                lossD = lossDreal + lossDfake
                lossD.backward(retain_graph=False)
                optimizerD.step()
                
                
                #train D again
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                r_hat, D_hat_res_list = D(x_hat, e_finetuning=e_hat, i=0)
                r, D_res_list = D(x, e_finetuning=e_hat, i=0)
    
                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)
    
                lossD = lossDreal + lossDfake
                lossD.backward(retain_graph=False)
                optimizerD.step()
    
    
            # Output training stats
            # if i_batch % 10 == 0:
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

        # if display_training:
        #         plt.clf()
        #         out = (x_hat[0]*255).transpose(0,2)
        #         for img_no in range(1,x_hat.shape[0]):
        #             out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 1)
        #         out = out.type(torch.int32).to(cpu).numpy()
        #         fig = out

        #         plt.clf()
        #         out = (x[0]*255).transpose(0,2)
        #         for img_no in range(1,x.shape[0]):
        #             out = torch.cat((out, (x[img_no]*255).transpose(0,2)), dim = 1)
        #         out = out.type(torch.int32).to(cpu).numpy()
        #         fig = np.concatenate((fig, out), 0)

        #         plt.clf()
        #         out = (g_y[0]*255).transpose(0,2)
        #         for img_no in range(1,g_y.shape[0]):
        #             out = torch.cat((out, (g_y[img_no]*255).transpose(0,2)), dim = 1)
        #         out = out.type(torch.int32).to(cpu).numpy()
                
        #         fig = np.concatenate((fig, out), 0)
        #         plt.imshow(fig)
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.draw()
        #         plt.pause(0.001)
    
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
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        }, path_to_save)
print('...Done saving latest')
