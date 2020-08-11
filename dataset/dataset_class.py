import torch
from torch.utils.data import Dataset
import os
import numpy as np
import face_alignment
import bisect
import albumentations as A

from .video_extraction_conversion import *

augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=.2, contrast_limit=.2),
    A.RandomGamma(gamma_limit=(20,200)),
    A.CLAHE(),
    A.Blur(),
    A.JpegCompression(quality_lower=5),
], p=1)

class VidDataSet(Dataset):
    def __init__(self, K, path_to_mp4, device):
        self.K = K
        self.path_to_mp4 = path_to_mp4
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')
    
    def __len__(self):
        vid_num = 0
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):
                for video in os.listdir(os.path.join(self.path_to_mp4, person_id, video_id)):
                    vid_num += 1
        return vid_num
    
    def __getitem__(self, idx):
        vid_idx = idx
        if idx<0:
            idx = self.__len__() + idx
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):
                for video in os.listdir(os.path.join(self.path_to_mp4, person_id, video_id)):
                    if idx != 0:
                        idx -= 1
                    else:
                        break
                if idx == 0:
                    break
            if idx == 0:
                break
        path = os.path.join(self.path_to_mp4, person_id, video_id, video)
        frame_mark = select_frames(path , self.K)
        frame_mark = generate_landmarks(frame_mark, self.face_aligner)
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3
        frame_mark = frame_mark.transpose(2,4).to(self.device)/255 #K,2,3,224,224

        g_idx = torch.randint(low = 0, high = self.K, size = (1,1))
        x = frame_mark[g_idx,0].squeeze()
        g_y = frame_mark[g_idx,1].squeeze()
        return frame_mark, x, g_y, vid_idx

class PreprocessDataset(Dataset):
    def __init__(self, K, path_to_images, path_to_segs, path_to_Wi):
        self.K = K
        self.path_to_images = path_to_images
        self.path_to_segs = path_to_segs
        self.path_to_Wi = path_to_Wi
        
        self.person_id_list = [(i, len(os.listdir(os.path.join(self.path_to_images, i))))\
             for i in sorted(os.listdir(self.path_to_images))]
        
        self.vid_num = np.cumsum([i for _, i in self.person_id_list])
        
    def __len__(self):
        return self.vid_num[-1]
    
    def __getitem__(self, idx):
        # print('STARTED!')
        if idx<0:
            idx = self.__len__() + idx - 1
        
        person_idx = np.searchsorted(self.vid_num, idx, side='right')
        person_id, num_frames = self.person_id_list[person_idx]
        start_idx = self.vid_num[person_idx - 1] if person_idx > 0 else 0
        internal_idx = idx - start_idx
        repeats = 1 + (self.K - 1)//(num_frames - 1)
        idxs = list(range(0, internal_idx)) + list(range(internal_idx + 1, num_frames))
        # print(person_id, num_frames, idxs)
        if repeats > 1:
            idxs = np.tile(idxs, repeats)
        identity_idxs = np.random.choice(
            a=idxs,
            size=self.K,
            replace=False)
        img_path_list = [os.path.join(
            self.path_to_images,
            person_id,
            '{:07d}.png'.format(i)) for i in identity_idxs]
        identity_imgs = select_preprocess_frames(
            img_path_list=img_path_list,
            seg_path_list=[],)[0]
        identity_imgs = torch.from_numpy(np.array(identity_imgs)).type(dtype = torch.float) #K,256,256,3
        identity_imgs = identity_imgs.permute(0,3,1,2)/255 #K,3,256,256
        img_path_list = [os.path.join(
            self.path_to_images,
            person_id,
            '{:07d}.png'.format(internal_idx))]
        seg_path_list = [os.path.join(
            self.path_to_segs,
            person_id,
            '{:07d}.png'.format(internal_idx))]
        pose_img, pose_seg = select_preprocess_frames(
            img_path_list=img_path_list,
            seg_path_list=seg_path_list,)
        
        pose_aug = augment(image=pose_img[0])['image']
        pose_aug = torch.from_numpy(pose_aug).type(dtype = torch.float)
        pose_aug = pose_aug.permute(2,0,1)/255
        
        pose_img = torch.from_numpy(np.array(pose_img[0])).type(dtype = torch.float) #256,256,3
        pose_img = pose_img.permute(2,0,1)/255 #3,256,256
        
        pose_seg = torch.from_numpy(np.array(pose_seg)).type(dtype = torch.int) #1,256,256
        if self.path_to_Wi is not None:
            try:
                W_i = torch.load(self.path_to_Wi+'/W_'+str(idx)+'/W_'+str(idx)+'.tar',
                            map_location='cpu')['W_i'].requires_grad_(False)
            except:
                print("\n\nerror loading: ", self.path_to_Wi+'/W_'+str(idx)+'/W_'+str(idx)+'.tar')
                W_i = torch.rand((512,1))
        else:
            W_i = None
        return identity_imgs, pose_img, pose_aug, pose_seg, idx, W_i
    
# class PreprocessDataset(Dataset):
#     def __init__(self, K, path_to_preprocess, path_to_Wi):
#         self.K = K
#         self.path_to_preprocess = path_to_preprocess
#         self.path_to_Wi = path_to_Wi
        
#         self.person_id_list = os.listdir(self.path_to_preprocess)
#     def __len__(self):
#         vid_num = 0
#         for person_id in self.person_id_list:
#             vid_num += len(os.listdir(os.path.join(self.path_to_preprocess, person_id)))
#         return vid_num-1
    
#     def __getitem__(self, idx):
#         vid_idx = idx
#         if idx<0:
#             idx = self.__len__() + idx
#         path = os.path.join(self.path_to_preprocess,
#                             str(idx//256),
#                             str(idx)+".png")
#         frame_mark = select_preprocess_frames(path)
#         frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3
#         frame_mark = frame_mark.transpose(2,4)/255 #K,2,3,224,224

#         g_idx = torch.randint(low = 0, high = self.K, size = (1,1))
#         x = frame_mark[g_idx,0].squeeze()
#         g_y = frame_mark[g_idx,1].squeeze()
        
#         if self.path_to_Wi is not None:
#             try:
#                 W_i = torch.load(self.path_to_Wi+'/W_'+str(vid_idx)+'/W_'+str(vid_idx)+'.tar',
#                             map_location='cpu')['W_i'].requires_grad_(False)
#             except:
#                 print("\n\nerror loading: ", self.path_to_Wi+'/W_'+str(vid_idx)+'/W_'+str(vid_idx)+'.tar')
#                 W_i = torch.rand((512,1))
#         else:
#             W_i = None
        
#         return frame_mark, x, g_y, vid_idx, W_i


class FineTuningImagesDataset(Dataset):
    def __init__(self, path_to_images, device):
        self.path_to_images = path_to_images
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')
    
    def __len__(self):
        return len(os.listdir(self.path_to_images))
    
    def __getitem__(self, idx):
        frame_mark_images = select_images_frames(self.path_to_images)
        random_idx = torch.randint(low = 0, high = len(frame_mark_images), size = (1,1))
        frame_mark_images = [frame_mark_images[random_idx]]
        frame_mark_images = generate_cropped_landmarks(frame_mark_images, self.face_aligner, pad=50)
        frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark_images = frame_mark_images.transpose(2,4).to(self.device) #1,2,3,256,256
        
        x = frame_mark_images[0,0].squeeze()/255
        g_y = frame_mark_images[0,1].squeeze()/255
        
        return x, g_y
        

class FineTuningVideoDataset(Dataset):
    def __init__(self, path_to_video, device):
        self.path_to_video = path_to_video
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        path = self.path_to_video
        frame_has_face = False
        while not frame_has_face:
            try:
                frame_mark = select_frames(path , 1)
                frame_mark = generate_cropped_landmarks(frame_mark, self.face_aligner, pad=50)
                frame_has_face = True
            except:
                print('No face detected, retrying')
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark = frame_mark.transpose(2,4).to(self.device) #1,2,3,256,256
        
        x = frame_mark[0,0].squeeze()/255
        g_y = frame_mark[0,1].squeeze()/255
        return x, g_y
