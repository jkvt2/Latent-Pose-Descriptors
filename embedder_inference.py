"""Main"""
import torch
import os

from dataset.dataset_class import select_preprocess_frames
import torchvision.models as models

import numpy as np

from params.params import (
    path_to_chkpt, path_to_identity_embedding, path_to_identity_img)


"""Hyperparameters and config"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_e_hat_images = path_to_identity_embedding
path_to_i_images = path_to_identity_img

"""Loading Embedder input"""
img_path_list = [os.path.join(path_to_i_images, i) for i in sorted(os.listdir(path_to_i_images))]
identity_imgs = select_preprocess_frames(
    img_path_list=img_path_list,
    seg_path_list=[],)[0]
identity_imgs = torch.from_numpy(np.array(identity_imgs)).type(dtype = torch.float) #K,256,256,3
identity_imgs = identity_imgs.permute(0,3,1,2)/255 #K,3,256,256

Ei = models.resnext50_32x4d(num_classes=512).to(device)
Ei.eval()

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
Ei.load_state_dict(checkpoint['Ei_state_dict'])

"""Inference"""
with torch.no_grad():
    #forward
    ei_vectors = Ei(identity_imgs.to(device))
    print(ei_vectors.shape)


print('Saving e_hat...')
torch.save({
        'ei': ei_vectors.mean(dim=0),
        }, path_to_e_hat_images)
print('...Done saving')