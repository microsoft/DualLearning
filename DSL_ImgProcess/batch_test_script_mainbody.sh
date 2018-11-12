export PATH=/usr/anaconda2/bin:$PATH
#export LD_LIBRARY_PATH=~/Downloads/cuda/lib64:"$LD_LIBRAYR_PATH"
export CUDA_VISIBLE_DEVICES=6

model_dir=checkpoints

for (( e=345;e<=345;e+=2 ));do
filename=$(ls "$model_dir" | grep -o 'params_'${e}'uidx[^\.]*\.ckpt\.index')
filename=${filename:0:-6}

python monitor.py --data_dir=./cifar10_data --save_dir=$model_dir --batch_size=12 --show_interval=100 --load_params=${filename} --mode=I2L --useSoftLabel=0

done

for (( e=345;e<=345;e+=2 ));do
filename=$(ls "$model_dir" | grep -o 'params_'${e}'uidx[^\.]*\.ckpt\.index')
filename=${filename:0:-6}

python monitor.py --data_dir=./cifar10_data --save_dir=$model_dir --batch_size=12 --show_interval=100 --load_params=${filename} --mode=L2I --useSoftLabel=0

done


# Reminder: When using "--oneside" in training mode, you should also add the 
# corresponding "--oneside" in the inference phase.
