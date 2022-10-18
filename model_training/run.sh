export PYTHONPATH='./:./data:./clean_dataset:./Turk'


CUDA_VISIBLE_DEVICES=3 python train.py --dataset=face_attribute --noise_type=rand1 --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.0001 --batch_size=32

# Full dataset
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=48

# Debug dataset
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=48
