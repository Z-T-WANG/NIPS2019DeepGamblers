# NIPS2019DeepGamblers
This repository provides code to accompany NIPS2019 paper __Deep Gamblers: Learning to Abstain with Portfolio Theory__ https://arxiv.org/abs/1907.00208  

The code aims to provide an implementation of the method introduced and only supports vgg16 and vgg16_bn models unless manually modified (it defaults to vgg16_bn). 
   
## Use
To train models for rewards (payoffs) o1, o2, o3 respectively,     
  
```python3 main.py --rewards o1 o2 o3 --dataset cifar10/svhn/catsdogs```   
   
To evaluate the validation error and test error of the trained models with specified predicition coverages,  
  
```python3 main.py --rewards o1 o2 o3 --dataset cifar10/svhn/catsdogs --evaluate --coverage cov1 cov2...```   

In addition, `--save` argument can be used to specify a path to save trained models and evluate trained models, and `--pretrain` argument specifies how many epochs are used for pretraining with the conventional cross entropy loss. Pretraining is useful in case the learning does not start due to a low `o` parameter. `--epochs` defaults to 300. When `--dataset` is `cifar10`, `--pretrain` defaults to 100 if `o<6.1` and defaults to 0 otherwise.
   
   
The code is based on https://github.com/bearpaw/pytorch-classification
