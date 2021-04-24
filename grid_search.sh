#!/bin/bash


for((i=1;i<=10;i++));
do
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_bi_kl_epoch;
done