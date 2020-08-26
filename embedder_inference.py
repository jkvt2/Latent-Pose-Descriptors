"""Main"""
import torch
# import torch.nn as nn
import os

from dataset.video_extraction_conversion import select_preprocess_frames
# from network.blocks import *
import torchvision.models as models
# import face_alignment

import numpy as np

from params.params import path_to_chkpt


"""Hyperparameters and config"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
# path_to_e_hat_video = 'e_hat_video.tar'
path_to_e_hat_images = 'e_hat_images.tar'
# path_to_video = 'test_vid.mp4'
path_to_i_images = '/media/vince/storage/dl/data/identity_source/img'

# T = 32
# face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')


"""Loading Embedder input"""
# frame_mark_video = select_frames(path_to_video , T)
# frame_mark_video = generate_cropped_landmarks(frame_mark_video, pad=50, face_aligner=face_aligner)
# frame_mark_video = torch.from_numpy(np.array(frame_mark_video)).type(dtype = torch.float) #T,2,256,256,3
# frame_mark_video = frame_mark_video.transpose(2,4).to(device)/255 #T,2,3,256,256
# f_lm_video = frame_mark_video.unsqueeze(0) #1,T,2,3,256,256

# frame_mark_images = select_images_frames(path_to_images)
# frame_mark_images = generate_cropped_landmarks(frame_mark_images, pad=50, face_aligner=face_aligner)
# frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype = torch.float) #T,2,256,256,3
# frame_mark_images = frame_mark_images.transpose(2,4).to(device)/255 #T,2,3,256,256
# f_lm_images = frame_mark_images.unsqueeze(0) #1,T,2,3,256,256

img_path_list = [os.path.join(path_to_i_images, i) for i in sorted(os.listdir(path_to_i_images))]
identity_imgs = select_preprocess_frames(
    img_path_list=img_path_list,
    seg_path_list=[],)[0]
identity_imgs = torch.from_numpy(np.array(identity_imgs)).type(dtype = torch.float) #K,256,256,3
identity_imgs = identity_imgs.permute(0,3,1,2)/255 #K,3,256,256

# pose_img = torch.from_numpy(np.array(pose_img[0])).type(dtype = torch.float) #256,256,3
# pose_img = pose_img.permute(2,0,1)/255 #3,256,256

# pose_seg = torch.from_numpy(np.array(pose_seg)).type(dtype = torch.int) #1,256,256

Ei = models.resnext50_32x4d(num_classes=512).to(device)
Ei.eval()

# E = Embedder(256).to(device)
# E.eval()



"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
Ei.load_state_dict(checkpoint['Ei_state_dict'])




"""Inference"""
with torch.no_grad():
    #forward
    # Calculate average encoding vector for video
    # f_lm = f_lm_video
    # f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
    # e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
    # e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
    # e_hat_video = e_vectors.mean(dim=1)
    
    
    # f_lm = f_lm_images #1,T,2,3,256,256
    # f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
    # e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
    # e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
    # e_hat_images = e_vectors.mean(dim=1)
    ei_vectors = Ei(identity_imgs.to(device))
    print(ei_vectors.shape)


print('Saving e_hat...')
# torch.save({
#         'e_hat': e_hat_video
#         }, path_to_e_hat_video)
torch.save({
        'ei': ei_vectors.mean(dim=0),
        }, path_to_e_hat_images)
print('...Done saving')