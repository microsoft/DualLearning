export PATH=/usr/anaconda2/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# train two models (test 4 gpu)
python monitor.py --data_dir=./cifar10_data --save_dir=./checkpoints_All --batch_size=12 --show_interval=10 --learning_rate=1e-4 --load_params=params_345uidx480248.ckpt --learning_rate_I2L=2e-4 --trade_off_I2L=30. --trade_off_L2I=1.5 --save_interval=1 --bias=0.02 --valid_interval=8 --lr_decay=1. --nr_gpu=4

# train image classifier only (test single gpu)
# python monitor.py --data_dir=./cifar10_data --save_dir=./checkpoints_I2L --batch_size=12 --show_interval=10 --learning_rate=1e-4 --load_params=params_345uidx480248.ckpt --learning_rate_I2L=2e-4 --trade_off_I2L=30. --trade_off_L2I=1.5 --save_interval=1 --bias=0.02 --valid_interval=8 --lr_decay=1. --nr_gpu=1 --oneside=I2L

# train image generator only (test 2 gpu)
# python monitor.py --data_dir=./cifar10_data --save_dir=./checkpoints_L2I --batch_size=12 --show_interval=10 --learning_rate=1e-4 --load_params=params_345uidx480248.ckpt --learning_rate_I2L=2e-4 --trade_off_I2L=30. --trade_off_L2I=1.5 --save_interval=1 --bias=0.02 --valid_interval=8 --lr_decay=1. --nr_gpu=2 --oneside=L2I
