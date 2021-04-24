#!/bin/bash

for loss_efficient in 0.1 0.2 0.3 0.4
do
  for temperature in 2 3 4 5
  do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_bi_ensemble.py --loss_coefficient $loss_efficient --temperature $temperature
  done
done