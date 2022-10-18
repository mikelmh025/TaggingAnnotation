export PYTHONPATH='./:./data:./clean_dataset:./Turk'


CUDA_VISIBLE_DEVICES=3 python train.py --dataset=face_attribute --noise_type=rand1 --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.0001 --batch_size=32

# Full dataset
CUDA_VISIBLE_DEVICES=1,2 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=64 --print_freq=50 --n_epoch=300

# Debug dataset
CUDA_VISIBLE_DEVICES=2,3 nohup python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=16 --debug --print_freq=5


cp debug_dir/face_attribute/resnet/mselast.pth.tar debug_dir/