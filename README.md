# NIPS2019DeepGamblers
This repository provides code to accompany NIPS2019 paper __Deep Gamblers: Learning to Abstain with Portfolio Theory__ https://arxiv.org/abs/1907.00208  

The code aims to provide an implementation of the method introduced and only supports vgg16 and vgg16_bn models unless manually modified (it defaults to vgg16_bn). 
   
## Method  
Conventional deep learning image classification minimizes a cross entropy loss measured between the network prediction and the realistic training data. To evaluate the AI's uncertainty about its own prediction, we give it an additional prediction choice corresponding to abstention, and transform the original prediction problem to a gambling problem. The AI choose some prediction choices to bet on and additionally reserve a portion of its money on the abstention choice, and then the AI is trained to maximize the doubling rate of its money. This idea is inspired by portfolio theory.   

The training loss for a labelled data `(x,y)` is therefore  

```l(x,y)=-log(o*f(x)_y + f(x)_{m+1})```  
   
where `o` is the reward (payoff) of the prediction on label `y`, and there are `m` categories to choose from and `f(x)_{m+1}` is the prediction on abstention. `f(x)` is a distribution satisfying `\sum_i f(x)_i = 1` 
   
## Use  
To train models for correct prediction rewards (payoffs) o1, o2, o3 respectively,     
  
```python3 main.py --rewards o1 o2 o3 --dataset cifar10/svhn/catsdogs```   
   
To evaluate the validation error and test error of the trained models with specified predicition coverages,  
  
```python3 main.py --rewards o1 o2 o3 --dataset cifar10/svhn/catsdogs --evaluate --coverage cov1 cov2...```   

In addition, `--save` argument can be used to specify a path to save trained models and evluate them, and `--pretrain` argument specifies how many epochs are used for pretraining with the conventional cross entropy loss. Pretraining is useful in case the learning does not start due to a low `o` parameter. `--epochs` defaults to 300. When `--dataset` is `cifar10`, `--pretrain` defaults to 100 if `o<6.1` and defaults to 0 otherwise.
   
   
The code is based on https://github.com/bearpaw/pytorch-classification
