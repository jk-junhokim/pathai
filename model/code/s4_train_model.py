import os
import time
import wandb
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

import my_arguments
from my_arguments import ModelArgs
from my_utils import getNetwork
from my_model_trainer import TrainModel
from my_data_augment import AugmentData
from s1_param import MyHyperparmeter as hp

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
   
if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = ModelArgs().parse()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    net = getNetwork(args)
    
    wandb.init(project=f"{hp.DATASET_NAME}-7-18-Train")
    wandb.run.name = f"{hp.RUN_NAME}"
    wandb.watch(net)
    
    train_data = AugmentData(args, mode="train")
    trainloader = DataLoader(train_data,
                             batch_size=args.train_batch_size,
                             shuffle=True,
                             num_workers=args.nWorker)
    
    validation_data = AugmentData(args, mode="validation")
    valloader = DataLoader(validation_data,
                           batch_size=args.val_batch_size,
                           shuffle=False,
                           num_workers=args.nWorker)
    
    
    model = TrainModel(args, net)
    model.train(trainloader=trainloader, valloader=valloader)
    
"""
./s4_train_model
├── my_utils
│   │ 
│   └── getNetwork
│  
├── my_model_trainer
│   │   
│   └── TrainModel
│      │
|      └──train
|
└──  my_data_augment
    │   
    └── AugmentData      
"""



