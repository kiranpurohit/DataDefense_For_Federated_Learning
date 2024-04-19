## DataDefense: Mitigating Model Poisoning Attacks in Federated Learning

### Depdendencies (tentative)
---
Tested stable depdencises:
* python 3.6.5 (Anaconda)
* PyTorch 1.1.0
* torchvision 0.2.2
* CUDA 10.0.130
* cuDNN 7.5.1

### Data Preparation
---
1. For Southwest Airline (for CIFAR-10) edge-case example, the collected edge-case datasets are available in `./saved_datasets`. 
2. For Trigger Patch, datasets are available in `./saved_datasets`.  

### Running Experients:
---
The main script is `./simulated_averaging.py`, to launch the jobs, we provide a script `./run_simulated_averaging.sh`. And we provide a detailed description on the arguments.


| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `lr` | Inital learning rate that will be used for local training process. |
| `batch-size` | Batch size for the optimizers e.g. SGD. |
| `dataset`      | Dataset to use. |
| `model`      | Model to use. |
| `gamma` | the factor of learning rate decay, i.e. the effective learning rate is `lr*gamma^t`. |
| `batch-size` | Batch size for the optimizers e.g. SGD. |
| `num_nets` | The total number of available users e.g. 200 for CIFAR-10. |
| `fl_round` | maximum number of FL rounds for the code to run. |
| `part_nets_per_round` | Number of active users that are sampled per FL round to participate. |
| `local_train_period` | Number of local training epochs that the honest users can run. |
| `adversarial_local_training_period`  | Number of local training epochs that the attacker(s) can run. |
| `fl_mode`    | `fixed-freq` or `fixed-pool` for fixed frequency and fixed pool attacking settings.  |
| `attacker_pool_size`    | Number of attackers in the total number of available users, used only when `fl_mode=fixed-pool`. |
| `defense_method`    | Defense method over the data center end.   |
| `stddev` | Standard deviation of the noise added for weak DP defense. |
| `norm_bound` | Norm bound for the norm difference clipping defense. |
| `attack_method` | Attacking schemes used for attacker and either be `blackbox` or `PGD`. |
| `attack_case` | Wether or not to conduct edge-case attack, can be `edge-case`, `normal-case` or `almost-edge-case`. |
| `model_replacement` | Used when `attack_method=PGD` to control if the attack is PGD with replacement or without replacement. |
| `project_frequency` | How frequent (in how many iterations) to project to the l2 norm ball in PGD attack. |
| `eps` | Radius the l2 norm ball in PGD attack. |
| `adv_lr` | Learning rate of the attacker when conducting PGD attack. |
| `poison_type` | Specify the backdoor for each dataset using `southwest` for CIFAR-10. |
| `device` | Specify the hardware to run the experiment. |


#### Sample command
PGD Attack on Southwest Airline exmaple over CIFAR-10 dataset where there is defense on the data center. The attacker participate in the fixed-frequency manner.
```
python simulated_averaging.py \
--lr 0.001 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg9 \
--fl_mode fixed-freq \
--attacker_pool_size 10 \
--defense_method data-defense \
--attack_method pgd \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 1.5 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda
```

PGD Attack on Trigger Patch exmaple over CIFAR-10 dataset where there is defense on the data center. The attacker participate in the fixed-frequency manner.
```
python simulated_averaging.py \
--lr 0.001 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset triggerpatch \
--model vgg9 \
--fl_mode fixed-freq \
--attacker_pool_size 10 \
--defense_method data-defense \
--attack_method pgd \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 0.5 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type label_flip \
--norm_bound 2 \
--device=cuda
```
