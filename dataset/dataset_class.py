import torch
from torch.utils.data import Dataset
import os
import numpy as np
# import face_alignment
# import bisect
import albumentations as A
import albumentations.augmentations.functional as F
import cv2
import random

from .video_extraction_conversion import select_preprocess_frames

def scale(img, scale_x, scale_y, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale_y), int(width * scale_x)
    return F.resize(img, new_height, new_width, interpolation)

class RandomIndependentScale(A.DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.

    Args:
        scale_limit_x ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (1 - scale_limit, 1 + scale_limit). Default: (0.9, 1.1).
        scale_limit_y ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (1 - scale_limit, 1 + scale_limit). Default: (0.9, 1.1).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, scale_limit_x=0.1, scale_limit_y=0.1, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomIndependentScale, self).__init__(always_apply, p)
        self.scale_limit_x = A.to_tuple(scale_limit_x, bias=1.0)
        self.scale_limit_y = A.to_tuple(scale_limit_y, bias=1.0)
        self.interpolation = interpolation

    def get_params(self):
        return {"scale_x": random.uniform(self.scale_limit_x[0], self.scale_limit_x[1]),
                "scale_y": random.uniform(self.scale_limit_y[0], self.scale_limit_y[1])}

    def apply(self, img, scale_x=0, scale_y=0, interpolation=cv2.INTER_LINEAR, **params):
        return scale(img, scale_x, scale_y, interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, scale_x=0, scale_y=0, **params):
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation,
                "scale_limit_x": A.to_tuple(self.scale_limit_x, bias=-1.0),
                "scale_limit_y": A.to_tuple(self.scale_limit_y, bias=-1.0)}

augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=.2, contrast_limit=.2),
    A.RandomGamma(gamma_limit=(20,200)),
    A.CLAHE(),
    A.Blur(),
    A.JpegCompression(quality_lower=5),
    RandomIndependentScale(scale_limit_x=.2, scale_limit_y=.2),
    A.PadIfNeeded(min_height=256,min_width=256),
    A.CenterCrop(height=256,width=256),
], p=1)

class PreprocessDataset(Dataset):
    def __init__(self, K, path_to_images, path_to_segs, path_to_Wi):
        self.K = K
        self.path_to_images = path_to_images
        self.path_to_segs = path_to_segs
        self.path_to_Wi = path_to_Wi
        
        self.person_id_list = [(i, len(os.listdir(os.path.join(self.path_to_segs, i))))\
             for i in sorted(os.listdir(self.path_to_segs))]
        
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
        
        pose_seg = torch.from_numpy(np.array(pose_seg)).type(dtype = torch.float) #1,256,256
        if self.path_to_Wi is not None:
            wi_path = self.path_to_Wi+'/W_'+str(idx//256)+'/W_'+str(idx)+'.tar'
            try:
                W_i = torch.load(wi_path,
                            map_location='cpu')['W_i'].requires_grad_(False)
            except:
                print("\n\nerror loading: ", wi_path)
                W_i = torch.rand((768,1))
        else:
            W_i = None
        return identity_imgs, pose_img, pose_aug, pose_seg, idx, W_i
    
class FineTuningImagesDataset(Dataset):
    def __init__(self, path_to_images, path_to_segs, device):
        self.path_to_images = path_to_images
        self.path_to_segs = path_to_segs
        self.device = device
    
    def __len__(self):
        return len(os.listdir(self.path_to_images))
    
    def __getitem__(self, idx):

        img_path_list = [os.path.join(
            self.path_to_images,
            '{:07d}.png'.format(idx))]
        seg_path_list = [os.path.join(
            self.path_to_segs,
            '{:07d}.png'.format(idx))]
        pose_img, pose_seg = select_preprocess_frames(
            img_path_list=img_path_list,
            seg_path_list=seg_path_list,)
        
        pose_aug = augment(image=pose_img[0])['image']
        pose_aug = torch.from_numpy(pose_aug).type(dtype = torch.float)
        pose_aug = pose_aug.permute(2,0,1)/255
        
        pose_img = torch.from_numpy(np.array(pose_img[0])).type(dtype = torch.float) #256,256,3
        pose_img = pose_img.permute(2,0,1)/255 #3,256,256
        
        pose_seg = torch.from_numpy(np.array(pose_seg)).type(dtype = torch.int) #1,256,256
        
        return pose_aug, pose_img, pose_seg
        
