# Realistic-Neural-Talking-Head-Models

My implementation of Neural Head Reenactment with Latent Pose Descriptors (Egor Burkov et al.). https://arxiv.org/pdf/2004.12000
Forked from https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models.


Source video:

![](examples/source.gif)

Driving and output video:

![](examples/demo.gif)

Steps:

0. modify params/params.py as required
  - get VGGFace pretrained model according to prerequisites section below
  - get VGG19 pretrained model from default torchvision models
  - for the demo, you can use a video: specify it with path_to_pose_video and the images will be automatically cropped from it
1. run train.py to get model_weights.tar (the meta-learning model)
2. run embedder_inference.py (requires model_weights.tar) to get e_hat_images.tar (the embedding of the identity images)
3. run finetuning_training.py (requires model_weights.tar, e_hat_images.tar) to get finetuned_model.tar (the finetuned model)
4. run demo.py (requires finetuned_model.tar, e_hat_images.tar) to get gif.gif in vis/

## Prerequisites

### 1.Loading and converting the caffe VGGFace model to pytorch for the content loss:
Follow these instructions to install the VGGFace from the paper (https://arxiv.org/pdf/1703.07332.pdf):

```
$ wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz
$ tar xvzf vgg_face_caffe.tar.gz
$ sudo apt install caffe-cuda
$ pip install mmdnn
```

Convert Caffe to IR (Intermediate Representation)

`$ mmtoir -f caffe -n vgg_face_caffe/VGG_FACE_deploy.prototxt -w vgg_face_caffe/VGG_FACE.caffemodel -o VGGFACE_IR`

**If you have a problem with pickle, delete your numpy and reinstall numpy with version 1.16.1**

IR to Pytorch code and weights

`$ mmtocode -f pytorch -n VGGFACE_IR.pb --IRWeightPath VGGFACE_IR.npy --dstModelPath Pytorch_VGGFACE_IR.py -dw Pytorch_VGGFACE_IR.npy`

Pytorch code and weights to Pytorch model

`$ mmtomodel -f pytorch -in Pytorch_VGGFACE_IR.py -iw Pytorch_VGGFACE_IR.npy -o Pytorch_VGGFACE.pth`


At this point, you will have a few files in your directory. To save some space you can delete everything and keep **Pytorch_VGGFACE_IR.py** and **Pytorch_VGGFACE.pth**
### 2.Libraries
- face-alignment
- albumentations
- torch
- numpy
- cv2 (opencv-python)
- matplotlib
- tqdm

### 3.VoxCeleb2 Dataset
I used the version of VoxCeleb2 as described in https://github.com/AliaksandrSiarohin/video-preprocessing. Note that this is substantially different from the origninal paper, which centers the face at every frame. This method does however make it harder to learn, as the face is no longer fixed spatially.
