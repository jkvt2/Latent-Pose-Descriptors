#K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape, path_to_mp4

#number of frames to load
K = 8

#path to main weight
path_to_chkpt = 'model_weights.tar' 

#path to backup
path_to_backup = 'backup_model_weights.tar'

#CHANGE if better gpu
batch_size = 2

#CHANGE first part
path_to_Wi = ""+"Wi_weights"
#path_to_Wi = "test/"+"Wi_weights"

#dataset save path
path_to_images = '/media/hdd1/vince/data/lpd/img/train'
path_to_segs = '/media/hdd1/vince/data/lpd/seg/train'

# path_to_images = '/media/hdd1/vince/data/lpd/sample_img'
# path_to_segs = '/media/hdd1/vince/data/lpd/sample_seg'

path_to_images = '/media/vince/storage/dl/data/voxceleb/reorg'
path_to_segs = '/media/vince/storage/dl/data/voxceleb/seg'

#default for Voxceleb
frame_shape = 256