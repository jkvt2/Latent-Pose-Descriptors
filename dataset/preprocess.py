import numpy as np
import imageio
import os
from .util import bb_intersection_over_union, join, crop_bbox_from_frames
from skimage.transform import resize
import face_alignment

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

REF_FRAME_SIZE = 360
REF_FPS = 25

def extract_bbox(frame, refbbox=None):
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if refbbox is None:
        return bboxes
    else:
        if len(bboxes) != 0:
            bbox = max([(bb_intersection_over_union(bbox, refbbox), tuple(bbox)) for bbox in bboxes])[1]
        else:
            bbox = np.array([0, 0, 0, 0, 0])
        return np.maximum(np.array(bbox), 0)

def crop_video(video_path,
               mode='longest',
               iou_with_initial=0.5,
               aspect_preserving=False,
               image_shape=(256,256),
               min_size=256, 
               min_frames=64,
               max_frames=1024,
               increase=0.1):
    """Makes the required cropped frames from a given video

    Args:
        video_path (str): where the video is
        mode (str): how to choose the subsequence to crop frames from. Should be one of:
            longest: take the longest subsequence that conforms to requirements
            start: take the subsequence starting from the first frame
            parts: split the video into multiple individually conforming subsequences
            Default: 'longest'.
        iou_with_initial (float): minimum iou with initial frame to continue
            Default: 0.5
        aspect_preserving (bool): whether the aspect should be preserved
            Default: False
        image_shape ((int, int)): shape of crop
            Default: (256, 256)
        min_size (int): minimum size of face required to keep sequence
            Default: 256
        min_frames (int): minimum number of frames for a subsequence to be kept
            Default: 64
        max_frames (int): maximum number of frames for a subsequence to be kept
            Default: 1024
        increase (float): amount of border gap to include around the face
            Default: 0.1

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    assert any([mode == i for i in ['longest', 'start', 'parts']])
    reader = imageio.get_reader(video_path)
    
    if mode == 'longest':
        initial_bboxs = []
        dones = []
        n_frames = []
        for i, frame in enumerate(reader):
            mult = frame.shape[0] / REF_FRAME_SIZE
            rs_frame = resize(frame, (REF_FRAME_SIZE, int(frame.shape[1] / mult)), preserve_range=True)
         
            bboxes = extract_bbox(rs_frame)
            if len(bboxes) == 0:
                bbox = np.array([0, 0, 0, 0, 0])
            else:
                bbox = mult * bboxes[0][:4]
            
            for j, (i_bbox, done) in enumerate(zip(initial_bboxs, dones)):
                if done: continue
                isec = bb_intersection_over_union(i_bbox, bbox)
                if isec < iou_with_initial or i - j >= max_frames:
                    n_frames[j] = i - j
                    dones[j] = True
            initial_bboxs.append(bbox)
            dones.append(False)
            n_frames.append(-1)
            
        for j, v in enumerate(n_frames):
            if v == -1:
                n_frames[j] = i - j + 1
        start = np.argmax(n_frames)
        dur = n_frames[start]
        
        frame_list = []
        tube_bbox = bbox
        for i, (frame, bbox) in enumerate(zip(reader, initial_bboxs[start:])):
            if i < start:continue
            if i >= start + dur:break
            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)
        out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox,
                                        min_frames=min_frames,
                                        image_shape=image_shape,
                                        min_size=min_size, 
                                        increase_area=increase,
                                        aspect_preserving=aspect_preserving)
        chunks_data = [(start, i, out, final_bbox)]
    else:
        initial_bbox = None
        start = 0
        tube_bbox = None
        frame_list = []
        chunks_data = []
        for i, frame in enumerate(reader):
            mult = frame.shape[0] / REF_FRAME_SIZE
            rs_frame = resize(frame, (REF_FRAME_SIZE, int(frame.shape[1] / mult)), preserve_range=True)
         
            bboxes = extract_bbox(rs_frame)
            if len(bboxes) == 0:continue
            bbox = mult * bboxes[0][:4]
            
            if initial_bbox is None:
                initial_bbox = bbox
                start = i
                tube_bbox = bbox
            
            isec = bb_intersection_over_union(initial_bbox, bbox)
            if isec < iou_with_initial or len(frame_list) >= max_frames:
                out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox,
                                                min_frames=min_frames,
                                                image_shape=image_shape,
                                                min_size=min_size, 
                                                increase_area=increase,
                                                aspect_preserving=aspect_preserving)
                chunks_data.append((start, i, out, final_bbox))
                if mode == 'start':
                    break
                initial_bbox = bbox
                start = i
                tube_bbox = bbox
                frame_list = []
            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)

    return chunks_data

def save_to_file(video_path, path_to_pose_img):
    chunks_data = crop_video(video_path, aspect_preserving=True)
    os.makedirs(path_to_pose_img)
    for start, end, frames, bbox in chunks_data:
        if frames:
            for idx, img in enumerate(frames):
                imageio.imsave(os.path.join(
                    path_to_pose_img,
                    '{:07d}.png'.format(idx)),
                    img)