# ELG
This repository is the code of https://arxiv.org/abs/2308.14104
Our code is built on the code of POMO[1] and Omni-VRP[2]. We provide the varying-scale and meta-trained models to reproduce the test results in the paper.  

## Test ELG* on VRPLIB[3, 5]

Under the ELG/CVRP folder, use the default settings in *config.yml*, run

```
python test_vrplib.py
```

## Train ELG* on CVRP

First, generate the validation sets by

```
python generate_data.py
```

Modify the *load_checkpoint* term in *config.yml* to None (i.e., *load_checkpoint*: ), and run

```
python train.py
```

## Test ELG* on TSPLIB[4]

Under the ELG/TSP folder, use the default settings in *config.yml*, and run

```  
python test_tsplib.py
```

## Train ELG* on TSP

First, generate the validation sets by

```
python generate_data.py
```

Modify the *load_checkpoint* term in *config.yml* to None (i.e., *load_checkpoint*: ), and run

```
python train.py
```

## Test ELG*-meta on Specified distributions

The rotation (1000, R) and explosion (1000,E) test sets in the paper are obtained from the source code of Omni-VRP[2]. 

Under the ELG/CVRP-meta folder, run

```
python test.py
```

The default config is to test on (1000, E) in the paper. You can change the explosion test file to rotation file for testing on (1000,R). 





Reference:

[1] Kwon, Y.-D.; Choo, J.; Kim, B.; Yoon, I.; Gwon, Y.; and Min, S. 2020. POMO: Policy optimization with multiple optima for reinforcement learning. In *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 21188–21198. Virtual.

[2] Zhou, J.;Wu, Y.; Song,W.; Cao, Z.; and Zhang, J. 2023. Towards omni-generalizable neural methods for vehicle routing problems. In *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 42769–42789. Honolulu, HI.

[3] Uchoa, E.; Pecin, D.; Pessoa, A.; Poggi, M.; Vidal, T.; and Subramanian, A. 2017. New benchmark instances for the capacitated vehicle routing problem. European Journal of Operational Research, 257(3): 845–858.

[4] Reinelt, G. 1991. TSPLIB - A traveling salesman problem library. ORSA Journal on Computing, 3(4): 376–384.

[5] Arnold, F.; Gendreau, M.; and S¨orensen, K. 2019. Efficiently solving very large-scale routing problems. Computers & Operations Research, 107: 32–42.
