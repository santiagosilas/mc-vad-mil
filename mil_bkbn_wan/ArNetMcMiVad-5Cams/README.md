

Commands on Socialab for the McMiVad Project

```
python main.py \ 
    --device 0 \
    --dataset_path /home/socialab/silas/arnet-backbone/Pets2009 \
    --dataset_name_camA 001  \
    --dataset_name_camB 002 \
    --feature_modal rgb 
    --feature_size 1024 \
    --Lambda 1_20 \
    --k 4 \
    --max_epoch 15 \
    --sample_size 2 \
    --save_every 15
```

Commands on Mac M1 for the McMiVad Project

```
python main.py --dataset_path /Volumes/SILAS-HD/ACAD-2024/inputs/arnet-backbone/Pets2009 --dataset_name_camA 001  --dataset_name_camB 002 --feature_modal rgb --feature_size 1024 --sample_size 4 --max_epoch 5 --device cpu
```



Commands on Socialab Machine for the Single Camera Project

```
python main.py --dataset_path /home/socialab/silas/arnet-backbone/Pets2009 --dataset_name 00X --feature_modal rgb --feature_size 1024 --sample_size 4 --max_epoch 5
```