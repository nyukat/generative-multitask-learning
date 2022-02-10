# Generative multitask learning mitigates target-causing confounding

This repository contains the code necessary to reproduce our paper 
[Generative multitask learning mitigates target-causing confounding](https://arxiv.org/abs/2202.04136). 

## Setup

### Dependencies
* Python 3.9
* gin-config
* numpy
* matplotlib
* pandas
* pytorch
* scikit-learn
* scipy
* torchvision

### Datasets
* [Attributes of people](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/attributes_dataset.tgz)
* [Taskonomy](https://github.com/StanfordVL/taskonomy/tree/master/data)

### Workflow

This is the workflow necessary to reproduce the results in our paper. For example, this is the workflow for 
our Taskonomy experiment using cross-stitch networks, with object classification as the main task and scene classification 
as the auxiliary task. We used five random seeds in our paper, but in this example we will use one seed for simplicity.

**1. Train the single task networks**

python generative_multitask_learning/experiment.py config/taskonomy/stl/t=0/s=0.gin (object classification)

python generative_multitask_learning/experiment.py config/taskonomy/stl/t=1/s=0.gin (scene classification)

**2. Train the multitask network**

python generative_multitask_learning/experiment.py config/taskonomy/mtl/cross_stitch_resnet/mt=0/s=0.gin

**3. Predict on the test set with the multitask network (below is for alpha=0.1, do this for all alpha)**

python generative_multitask_learning/experiment.py config/taskonomy/mtl_eval/cross_stitch_resnet/mt=0/s=0/a=0.1.gin

**4. Estimate out-of-distribution accuracy with importance sampling**

python generative_multitask_learning/importance_sampling.py \\  
--main_task_idx 0 \\  
--y_pred_dpath results/taskonomy/mtl/cross_stitch_resnet/mt=0/s=0 \\  
--log_prior_fpath results/taskonomy/prior/log_prior.pt \\  
--n_classes_list 100 64 \\  
--n_seeds 1
