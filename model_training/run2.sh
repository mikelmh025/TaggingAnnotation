CUDA_VISIBLE_DEVICES=0,1,2 nohup python train.py --seed=0 --result_dir=train_round2  --model=resnet50 \
    --batch_size=24 --loss=mse --lr=0.001  --print_freq=50 --n_epoch=300 \
    --num_workers=20 --multi_gpu \
    --dataset=face_attribute --target_mode=tag

CUDA_VISIBLE_DEVICES=0,1,2 nohup python train.py --seed=0 --result_dir=train_round2  --model=vit \
    --batch_size=6 --loss=mse --lr=0.001  --print_freq=50 --n_epoch=300 \
    --num_workers=20 --multi_gpu \
    --dataset=face_attribute --target_mode=tag