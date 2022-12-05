# taggingAnnotation

## Model training

Training the model using resnet. 
```
# Set path
export PYTHONPATH='./:./data:./data_processing'

# Train with Tags. The model predict the tags and use search algorithms for final output
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse --model=resnet --lr=0.00001 --batch_size=128 --print_freq=50 --n_epoch=300 --num_workers=20

# Train with direct annotation.  The model now treat the task as a classification task
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=face_attribute --seed=0 --result_dir=debug_dir --loss=mse  --model=resnet --lr=0.0001 --batch_size=32 --print_freq=50 --n_epoch=300 --num_workers=20 --target_mode=img

```

## Test the trained model

```
# Testing with Tag models
# Please modify checkpoint path
CUDA_VISIBLE_DEVICES=0 python test.py  --seed=0 --result_dir=test_results --model=resnet50 \
    --checkpoint=train_round2/face_attribute/resnet50/msebest.pt \
    --dataset=face_attribute --batch_size=24 --loss=mse --get_top_k=5 --target_mode=tag 
```

```
# Testing with direct models
# Please modify checkpoint path
CUDA_VISIBLE_DEVICES=0 python test.py  --seed=0 --result_dir=test_results --model=resnet50 \
    --checkpoint=train_round2/face_attribute/resnet50/msebest.pt \
    --dataset=face_attribute --batch_size=24 --loss=mse --get_top_k=5 --target_mode=img 
```

## Coming soon
1. Pretrained model
2. Human dataset: images, Labels
3. Asset images: Bitmojit, Google Cartoon Set, MetaHuman, NovelAI. 
4. Asset Labels: Bitmojit, Google Cartoon Set, MetaHuman, NovelAI.
5. Guide of using the search algorithm 


