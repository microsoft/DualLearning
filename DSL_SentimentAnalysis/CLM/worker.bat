@echo off
setlocal ENABLEDELAYEDEXPANSION
set THEANO_FLAGS=device=gpu1
python train_clm_WithDropout_lr0.5.py