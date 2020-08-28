###########################
#####COMMON PARAMETERS#####
###########################
frame_shape = 256
VGG19_weight_path = 'vgg19-dcbb9e9d.pth'
VGGFace_body_path='Pytorch_VGGFACE_IR.py'
VGGFace_weight_path='Pytorch_VGGFACE.pth'


###########################
####TRAINING PARAMETERS####
###########################
#number of frames to load
K = 8
batch_size = 2

#paths to checkpoints
path_to_chkpt = 'model_weights.tar' 
path_to_backup = 'backup_model_weights.tar'

#path to projection discrim
path_to_Wi = "Wi_weights"

#path to datasets
# path_to_images = '/media/hdd1/vince/data/lpd/img/train'
# path_to_segs = '/media/hdd1/vince/data/lpd/seg/train'
# path_to_images = '/media/hdd1/vince/data/lpd/sample_img'
# path_to_segs = '/media/hdd1/vince/data/lpd/sample_seg'
path_to_images = '/media/vince/storage/dl/data/voxceleb/reorg'
path_to_segs = '/media/vince/storage/dl/data/voxceleb/seg'


###########################
######DEMO PARAMETERS######
###########################
finetuning_batch_size = 2

path_to_finetuned_model = 'finetuned_model.tar'
path_to_identity_embedding = 'e_hat_images.tar'

#path to datasets
path_to_identity_img = '/media/vince/storage/dl/data/identity_source/img'
path_to_identity_seg = '/media/vince/storage/dl/data/identity_source/seg'
path_to_pose_img = '/media/vince/storage/dl/data/pose_source/img'
path_to_pose_video = '/media/vince/storage/dl/data/example.mp4'