
for wd in 0.0001 0.0002 0.0003 0.0004 0.0005; do

    CUDA_VISIBLE_DEVICES=1 python train_tree.py --wd ${wd}

done

for wd in 0.0001 0.0002 0.0003 0.0004 0.0005; do

    CUDA_VISIBLE_DEVICES=1 python train_origin.py --wd ${wd}

done
