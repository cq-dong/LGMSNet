
CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/isic18   --train_file_dir isic18_train1.txt --val_file_dir isic18_val1.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500
CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/isic18   --train_file_dir isic18_train2.txt --val_file_dir isic18_val2.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500
CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/isic18   --train_file_dir isic18_train3.txt --val_file_dir isic18_val3.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500

CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/busi   --train_file_dir busi_train1.txt --val_file_dir busi_val1.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500
CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/busi   --train_file_dir busi_train2.txt --val_file_dir busi_val2.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500
CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/busi   --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500

CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/tnscui   --train_file_dir tnscui_train1.txt --val_file_dir tnscui_val1.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500
CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/tnscui   --train_file_dir tnscui_train2.txt --val_file_dir tnscui_val2.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500
CUDA_VISIBLE_DEVICES=0 python main.py        --model SegmsNet      --base_dir ./data/tnscui   --train_file_dir tnscui_train3.txt --val_file_dir tnscui_val3.txt --base_lr 0.02 --seed 41 --batch_size 16 --txtnum 81 --epoch 500


CUDA_VISIBLE_DEVICES=0 python main_Kvasir.py        --model SegmsNet                            --base_dir ./data/Kvasir   --train_file_dir train.txt --val_file_dir val.txt --base_lr 0.01 --epoch 300 --seed 41   --batch_size 8  --txtnum 1
CUDA_VISIBLE_DEVICES=0 python main_Kvasir.py        --model SegmsNet                            --base_dir ./data/Kvasir   --train_file_dir train.txt --val_file_dir val.txt --base_lr 0.01 --epoch 300 --seed 1   --batch_size 8  --txtnum 1
CUDA_VISIBLE_DEVICES=0 python main_Kvasir.py        --model SegmsNet                            --base_dir ./data/Kvasir   --train_file_dir train.txt --val_file_dir val.txt --base_lr 0.01 --epoch 300 --seed 123   --batch_size 8  --txtnum 1


CUDA_VISIBLE_DEVICES=0 python main3d.py --max_epochs 5000 --val_every 50 --sw_batch_size 4 --batch_size 1 --logdir SegmsNet3D   --model_name SegmsNet3D --optim_lr 1e-3 --lrschedule warmup_cosine --infer_overlap 0.5 

