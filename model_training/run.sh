export PYTHONPATH='./:./data:./clean_dataset:./Turk:../Turk/:../clean_dataset'
sudo /dev/sda1 /media/otter/

CUDA_VISIBLE_DEVICES=3 python train.py --dataset=face_attribute --noise_type=rand1 --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.0001 --batch_size=32

# Full dataset
CUDA_VISIBLE_DEVICES=1,2 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=128 --print_freq=50 --n_epoch=300 --num_workers=20

# Debug dataset
CUDA_VISIBLE_DEVICES=2,3 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=16 --debug --print_freq=5


cp debug_dir/face_attribute/resnet/mselast.pth.tar debug_dir/


CUDA_VISIBLE_DEVICES=1,2 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse  --model=resnet --lr=0.0001 --batch_size=128 --print_freq=50 --n_epoch=300 --num_workers=20 --target_mode=img
CUDA_VISIBLE_DEVICES=0         python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse2 --model=resnet --lr=0.001 --batch_size=64 --print_freq=50 --n_epoch=300 --num_workers=20 --target_mode=img
