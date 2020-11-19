#CUDA_VISIBLE_DEVICES=0 python train.py --dataroot train_images --model self --dataset_mode single --name resnet --netG resnet_6blocks --no_flip --n_epochs 2000
#CUDA_VISIBLE_DEVICES=2 python train.py --dataroot train_images --model self --dataset_mode single --name resnet_direct --netG resnet_6blocks --no_flip --n_epochs 2000 --load_size 80 --crop_size 64
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot train_images --model self --dataset_mode single --name resnet_direct --netG resnet_6blocks --no_flip --n_epochs 2000 --load_size 80 --crop_size 64

#CUDA_VISIBLE_DEVICES=0 python get_features.py --dataroot train_images/all --model self --dataset_mode single --name resnet --netG resnet_6blocks
