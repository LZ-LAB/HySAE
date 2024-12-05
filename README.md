# The HySAE Model
This paper has been submitted to The Web Conference 2025 (WWW 2025).



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















































# Experimental Comparison
## Mixed Arity Knowledge Hypergraph


#### **1. JF17K**
| MODEL |        JF17K       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.580       |    0.478       |    0.610     |     0.729      |
| HJE     |     0.582       |    0.507       |    0.615     |     0.730      |
| HySAE   |     0.596       |    0.521       |    0.628     |     0.742      |

#### **2. WikiPeople**
| MODEL |        WikiPeople       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE   |     0.362       |      0.275         |     0.388       |   0.501      |
| HJE       |     0.444       |      0.368         |     0.485       |   0.577      |
| HySAE     |     0.454       |      0.373         |     0.496       |   0.603      |

#### **3. FB-AUTO**
| MODEL |        FB-AUTO       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.847         |     0.820        |    0.872        |     0.901      |
| HJE     |     0.871         |     0.849        |    0.883        |     0.909      |
| HySAE   |     0.893         |     0.876        |    0.904        |     0.926      |








## Fixed Arity Knowledge Hypergraph
### **1. WikiPeople-3**
| MODEL |        WikiPeople-3       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE   |     0.320          |     0.257    |    0.389      |     0.498      |
| HJE       |     0.373          |     0.279    |    0.403      |     0.528      |
| HySAE     |     0.389          |     0.297    |    0.420      |     0.578      |



### **2. JF17K-4**
| MODEL |        JF17K-4       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE   |     0.823          |     0.770    |    0.860      |     0.922      |
| HJE       |     0.817          |     0.763    |    0.854      |     0.918      |
| HySAE     |     0.834          |     0.780    |    0.871      |     0.932      |



### **3. WikiPeople-4**
| MODEL |        WikiPeople-4       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE   |     0.375          |     0.259    |    0.448      |     0.587      |
| HJE       |     0.321          |     0.195    |    0.390      |     0.576      |
| HySAE     |     0.410          |     0.289    |    0.481      |     0.639      |



### **4. JF17K-5**
| MODEL |        JF17K-5       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE   |     0.870          |     0.802    |    0.925      |     0.979      |
| HJE       |     0.791          |     0.722    |    0.873      |     0.930      |
| HySAE     |     0.887          |     0.828    |    0.941      |     0.985      |












## Performance Breakdown
### 1. **2-ary**
| MODEL |        JF17K       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.362       |    0.269       |    0.395     |     0.550      |
| HJE     |     0.364       |    0.271       |    0.397     |     0.556      |
| HySAE   |     0.389       |    0.295       |    0.419     |     0.583      |

| MODEL |        WikiPeople       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.374       |    0.304       |    0.457     |     0.510      |
| HJE     |     0.446       |    0.392       |    0.507     |     0.589      |
| HySAE   |     0.473       |    0.396       |    0.516     |     0.608      |

| MODEL |        FB-AUTO       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.400       |    0.329       |    0.430     |     0.528      |
| HJE     |     0.516       |    0.445       |    0.548     |     0.644      |
| HySAE   |     0.606       |    0.545       |    0.641     |     0.722      |




### 2. **3-ary**
| MODEL |        JF17K       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.594       |    0.526       |    0.615     |     0.723      |
| HJE     |     0.616       |    0.543       |    0.649     |     0.757      |
| HySAE   |     0.628       |    0.556       |    0.662     |     0.773      |

| MODEL |        WikiPeople       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.336       |    0.252       |    0.350     |     0.500      |
| HJE     |     0.348       |    0.260       |    0.376     |     0.506      |
| HySAE   |     0.352       |    0.263       |    0.380     |     0.535      |



### 3. **4-ary**
| MODEL |        JF17K       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.764       |    0.712       |    0.798     |     0.858      |
| HJE     |     0.795       |    0.744       |    0.826     |     0.891      |
| HySAE   |     0.809       |    0.756       |    0.842     |     0.903      |

| MODEL |        WikiPeople       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.309       |    0.208       |    0.322     |     0.526      |
| HJE     |     0.343       |    0.241       |    0.394     |     0.543      |
| HySAE   |     0.386       |    0.265       |    0.449     |     0.626      |

| MODEL |        FB-AUTO       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.457       |    0.369       |    0.500     |     0.619      |
| HJE     |     0.487       |    0.414       |    0.511     |     0.642      |
| HySAE   |     0.509       |    0.449       |    0.534     |     0.688      |



### 4. **5-ary**
| MODEL |        JF17K       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.786       |    0.701       |    0.861     |     0.896      |
| HJE     |     0.872       |    0.809       |    0.922     |     0.979      |
| HySAE   |     0.809       |    0.756       |    0.842     |     0.903      |

| MODEL |        WikiPeople       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.309       |    0.208       |    0.322     |     0.526      |
| HJE     |     0.343       |    0.241       |    0.394     |     0.543      |
| HySAE   |     0.386       |    0.265       |    0.449     |     0.626      |

| MODEL |        FB-AUTO       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.457       |    0.369       |    0.500     |     0.619      |
| HJE     |     0.487       |    0.414       |    0.511     |     0.642      |
| HySAE   |     0.509       |    0.449       |    0.534     |     0.688      |



### 5. **6-ary**
| MODEL |        JF17K       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.873       |    0.855       |    0.886     |     0.907      |
| HJE     |     0.893       |    0.864       |    0.916     |     0.948      |
| HySAE   |     0.956       |    0.938       |    0.969     |     0.979      |

| MODEL |        WikiPeople       |               |               |               |
|-------|--------------------|---------------|---------------|---------------|
|       | **MRR** | **Hit@1** | **Hit@3** | **Hit@10** |
| HyConvE |     0.275       |    0.187       |    0.281     |     0.470      |
| HJE     |     0.233       |    0.163       |    0.248     |     0.387      |
| HySAE   |     0.313       |    0.200       |    0.345     |     0.558      |




















## Model Efficiency

### **1. Model Parameters**
|  |               |               |               |
|-------|--------------------|---------------|---------------|
|   **Parameters (Millions)**    | **JF17K** | **WikiPeople** | **FB-AUTO** |
| HyConvE   |    12.80           |      21.44       |    4.80       |
| HJE       |        12.80          |    21.44     |      4.80     |
| HySAE     |       **1.38**       |     **2.34**    |    **1.06**   |



### **2. Memory Usage**
|  |               |               |               |
|-------|--------------------|---------------|---------------|
|   **Memory Usage (MB)**    | **JF17K** | **WikiPeople** | **FB-AUTO** |
| HyConvE   |    7718           |      15430       |    3032       |
| HJE       |      3322           |    4128     |      2470     |
| HySAE     |       **2712**           |     **3424**          |      **1692**    |




### **3. Time Usage**
|  |               |               |               |
|-------|--------------------|---------------|---------------|
|   **Time Usage (seconds)**    | **JF17K** | **WikiPeople** | **FB-AUTO** |
| HyConvE   |     98.7           |      247.3         |    4.9       |
| HJE       |       15.5          |    74.5     |   3.7      |
| HySAE     |       **10.1**           |     **34.1**          |      **2.2**         |











