export PYTHONPATH='./:./data:./clean_dataset:./Turk:../Turk/:../clean_dataset'
sudo mount /dev/sda1 /media/otter/
sudo du /media/ -sch

CUDA_VISIBLE_DEVICES=3 python train.py --dataset=face_attribute --noise_type=rand1 --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.0001 --batch_size=32

# Train with Tag
# Full dataset
CUDA_VISIBLE_DEVICES=1,2 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=128 --print_freq=50 --n_epoch=300 --num_workers=20

# Debug dataset
CUDA_VISIBLE_DEVICES=2,3 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=16 --debug --print_freq=5


cp debug_dir/face_attribute/resnet/mselast.pth.tar debug_dir/

# Train with Direct
CUDA_VISIBLE_DEVICES=1,2 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse  --model=resnet --lr=0.0001 --batch_size=32 --print_freq=50 --n_epoch=300 --num_workers=20 --target_mode=img
# Debug :direct
CUDA_VISIBLE_DEVICES=0         python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse  --model=resnet --lr=0.0001 --batch_size=16  --debug --print_freq=50 --n_epoch=300 --num_workers=2 --target_mode=img

# Resnet 34
CUDA_VISIBLE_DEVICES=1,2 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse  --model=resnet34 --lr=0.0001 --batch_size=32 --print_freq=50 --n_epoch=300 --num_workers=20 --target_mode=img


CUDA_VISIBLE_DEVICES=0         python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse  --model=vit      --lr=0.0001 --batch_size=4  --print_freq=50 --n_epoch=300 --num_workers=20 --target_mode=img
