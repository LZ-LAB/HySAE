# The HySAE Model
This paper has been submitted to The Web Conference 2025 (WWW 2025).

All local experiments were obtained on 4 NVIDIA GeForce RTX 3090 GPUs and PyTorch 1.12.0


## Requirements
The version of Python and major packages needed to run the code:
   
    -- python 3.9.16
    -- torch 1.12.0
    -- numpy 1.26.0
    -- tqdm 4.65.0



## How to Run

### HySAE-ISE

#### 1. Mixed Arity Knowledge Hypergraph
```
## JF17K dataset
python main-JF.py --dataset JF17K --batch_size 512 --lr 0.00086 --dr 0.993 --input_drop 0.0 --hidden_drop 0.4 --feature_drop 0.9 --k_size 1 --dil_size 2

## WikiPeople dataset
python main-WP.py --dataset WikiPeople --batch_size 512 --lr 0.00100 --dr 0.985 --input_drop 0.0 --hidden_drop 0.8 --feature_drop 0.2 --k_size 4 --dil_size 3

## FB-AUTO dataset
python main-FB.py --dataset FB-AUTO --batch_size 64 --lr 0.00049 --dr 0.979 --input_drop 0.8 --hidden_drop 0.2 --feature_drop 0.0 --k_size 3 --dil_size 3
```



#### 2. Fixed Arity Knowledge Hypergraph
```
## WikiPeople-3 dataset
python main-3ary.py --dataset WikiPeople-3 --batch_size 512 --lr 0.00094 --dr 0.967 --input_drop 0.5 --hidden_drop 0.6 --feature_drop 0.4 --k_size 1 --dil_size 2

## JF17K-4 dataset
python main-4ary.py --dataset JF17K-4 --batch_size 128 --lr 0.00040 --dr 0.980 --input_drop 0.4 --hidden_drop 0.5 --feature_drop 0.5 --k_size 3 --dil_size 1

## WikiPeople-4 dataset
python main-4ary.py --dataset WikiPeople-4 --batch_size 384 --lr 0.00021 --dr 0.997 --input_drop 0.7 --hidden_drop 0.3 --feature_drop 0.6 --k_size 1 --dil_size 2

## JF17K-5 dataset
python main-5ary.py --dataset JF17K-5 --batch_size 384 --lr 0.00072 --dr 0.999 --input_drop 0.0 --hidden_drop 0.6 --feature_drop 0.7 --k_size 1 --dil_size 2
```







### HySAE-ESE

#### 1. Mixed Arity Knowledge Hypergraph
```
## JF17K dataset
python main-JF.py --dataset JF17K --batch_size 512 --lr 0.00086 --dr 0.993 --input_drop 0.1 --hidden_drop 0.9 --feature_drop 0.2 --k_size 2 --k_sizeN 5 --dil_sizeN 5 --lamda 0.9

## WikiPeople dataset
python main-WP.py --dataset WikiPeople --batch_size 512 --lr 0.00083 --dr 0.999 --input_drop 0.0 --hidden_drop 0.5 --feature_drop 0.8 --k_size 1 --k_sizeN 1 --dil_sizeN 5 --lamda 0.5

## FB-AUTO dataset
python main-FB.py --dataset FB-AUTO --batch_size 64 --lr 0.00050 --dr 0.988 --input_drop 0.8 --hidden_drop 0.2 --feature_drop 0.3 --k_size 6 --k_sizeN 5 --dil_sizeN 4 --lamda 0.4
```



#### 2. Fixed Arity Knowledge Hypergraph
```
## WikiPeople-3 dataset
python main-3ary.py --dataset WikiPeople-3 --batch_size 64 --lr 0.00083 --dr 0.967 --input_drop 0.8 --hidden_drop 0.6 --feature_drop 0.2 --k_size 7 --k_sizeN 4 --dil_sizeN 5 --lamda 0.1

## JF17K-4 dataset
python main-4ary.py --dataset JF17K-4 --batch_size 256 --lr 0.00074 --dr 0.972 --input_drop 0.7 --hidden_drop 0.4 --feature_drop 0.2 --k_size 7 --k_sizeN 2 --dil_sizeN 5 --lamda 0.9

## WikiPeople-4 dataset
python main-4ary.py --dataset WikiPeople-4 --batch_size 128 --lr 0.00077 --dr 0.972 --input_drop 0.6 --hidden_drop 0.7 --feature_drop 0.6 --k_size 6 --k_sizeN 4 --dil_sizeN 3 --lamda 0.8

## JF17K-5 dataset
python main-5ary.py --dataset JF17K-5 --batch_size 512 --lr 0.00082 --dr 0.996 --input_drop 0.0 --hidden_drop 0.3 --feature_drop 0.9 --k_size 2 --k_sizeN 4 --dil_sizeN 4 --lamda 0.2
```


## Supplementary Note
To reproduce the results of the paper exactly, `torch.backends.cudnn.deterministic=True` in `main.py`.

However, due to problems with different versions of the Pytorch framework, this could seriously affect the speed of running the code.
If you encounter this problem, you can set `torch.backends.cudnn.deterministic=False` in `main.py` and it will restore the normal speed of our code.

Of course, this may lead to inconsistent results from each model run, but it will not affect the conclusions of the experiment. For example, in our open source code, `torch.backend.cudn.deterministic=False` is set by default on the `HySAE-E:JF17K` and `HySAE-E:WikiPeople` datasets to obtain the corresponding performance hyperparameters.




## Acknowledgments
We are very grateful for all open-source baseline models:

1. HypE/HSimplE: https://github.com/ElementAI/HypE
2. HyperMLN: https://github.com/zirui-chen/HyperMLN
3. RAM: https://github.com/liuyuaa/RAM
4. GETD: https://github.com/liuyuaa/GETD
5. tNaLP+: https://github.com/gsp2014/NaLP
6. PosKHG: https://github.com/zirui-chen/PosKHG
7. HyConvE: https://github.com/CarllllWang/HyConvE/tree/master
8. RD-MPNN: https://github.com/ooCher/RD-MPNN/tree/main/RD_MPNN
9. ReAlE: https://github.com/baharefatemi/ReAlE











