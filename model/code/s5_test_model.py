import os
import sys
import time
import wandb
from wandb.sklearn import plot_confusion_matrix as plt_cm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import my_arguments
from my_arguments import ModelArgs
from my_utils import getNetwork
from my_make_predictions import MakePredictions, EvalMetrics
from my_data_augment import AugmentData


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = ModelArgs().parse()
    
    if not args.is_pretrained:
        sys.exit()
    net = getNetwork(args)
    test_data = AugmentData(args, mode="test")
    testloader = DataLoader(test_data,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.nWorker)
    
    wandb.init(project=f"{my_arguments.DATASET_NAME} (Test)")
    wandb.watch(net)
    
    real, prediction = MakePredictions(testloader, net)
    result = EvalMetrics(real, prediction)
    
    for key in result:
        if key == 'Accuracy':
            print(key, ": {:.4f}".format(result[key]))
        if key == 'F1-Score':
            print(key, ": {:.4f}".format(result[key]))
    
    wandb.log({f'CM {my_arguments.DATASET_NAME}-PS-{my_arguments.PATCH_SIZE_STR}':
        plt_cm(y_true=real,
               y_pred=prediction,
               labels=[0, 1])})
    
    wandb.finish()
    np.savez(args.test_savename + f"/patch_{args.patch_size}_test_pred_score.npz",
             key_real=real,key_pred=prediction)
    print("############################\tModel Predictions Successfully Saved...")


    
