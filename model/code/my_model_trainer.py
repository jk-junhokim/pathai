import os
import time
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from my_make_predictions import MakePredictions, EvalMetrics
import my_arguments

class TrainModel:
    def __init__(self, args, network):
        self.args = args
        # self.network = network.cuda() if args.cuda else network
        if len(args.gpu_ids) > 1:
            self.network = torch.nn.DataParallel(network).cuda()
        else:
            self.network = network.cuda() if args.cuda else network
            
    def train(self, trainloader, valloader):
        
        self.trainloader = trainloader
        self.valloader = valloader
        print('####################\tLoading loss function and optimizer...')
        
        def AdjustLR(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
            for param_group in optimizer.param_groups:
                param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES, power), 8)
        
        loss_func = getattr(nn, self.args.loss)().cuda()
        
        
        
        if self.args.optimizer == "SGD":
            optimizer = optim.SGD(self.network.parameters(), lr=self.args.initLR, momentum=self.args.momentum, weight_decay=self.args.decay)
        elif self.args.optimizer == "Adam":
            optimizer = optim.Adam(self.network.parameters(), lr=self.args.initLR, betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.decay)
        elif self.args.optimizer == "RMSprop":
            optimizer = optim.RMSprop(self.network.parameters(), lr=self.args.initLR, momentum=self.args.momentum, weight_decay=self.args.decay)
        elif self.args.optimizer == "Adagrad":
            optimizer = optim.Adagrad(self.network.parameters(), lr=self.args.initLR, weight_decay=self.args.decay)
        else:
            raise ValueError("Invalid optimizer choice. Please select from SGD, Adam, RMSprop, or Adagrad.")
        
        
        
        best_acc = 0.0
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.05)
        for epoch in range(self.args.epoches):
            start = time.time()
            self.network.train()
            current_lr = optimizer.param_groups[0]['lr']
            AdjustLR(optimizer, epoch, 300, self.args.initLR, power=0.9)
            wandb.log({"Learning Rate": current_lr})

            losses = 0.0
            pbar = tqdm(total=len(self.trainloader))
            for i, (img, label, img_name) in enumerate(self.trainloader):
                
                if self.args.cuda:
                    img, label = img.cuda(), label.cuda()

                output = self.network(img) # size = (batch_size, 2)
                label = F.one_hot(label, num_classes=2).float() # for size compatibility
                loss = loss_func(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                
                pbar.set_description('Iteration {:3d}'.format(i + 1))
                pbar.set_postfix({'loss': '{:.4f}'.format(loss.item())})
                pbar.update()
            
            # scheduler.step()
            # AdjustLR(optimizer, epoch, self.args.epoches, self.args.initLR, power=0.9)
            pbar.close()      
            print('Epoch{:3d}\tTime(s){:.2f}\tAvgloss{:.4f}-'.format(epoch, time.time()-start, losses/(i+1)))
            wandb.log({"Time per Epoch": time.time()-start})
            wandb.log({"Average Loss": losses/(i+1)})

            real, prediction = MakePredictions(self.valloader, self.network)
            result = EvalMetrics(real, prediction)
            
            for key in result:
                if key == 'Accuracy':
                    print(key, ": {:.4f}".format(result[key]))
                    validation_acc = result[key]
                if key == 'F1-Score':
                    print(key, ": {:.4f}".format(result[key]))
            
            if validation_acc > best_acc:
                print('')
                print(f"Previous Best validation accuracy : {best_acc:.4f}")
                print(f"Best validation accuracy : {validation_acc:.4f}")
                print(f"Improve : {validation_acc - best_acc:.4f}")
                best_acc = validation_acc
                
                if not os.path.exists(self.args.train_savename):
                    os.makedirs(self.args.train_savename)
                    print("Directory created")
                
                torch.save(self.network.state_dict(),
                           self.args.train_savename + '/best_accuracy.pkl')
                print('Best model has been saved!')
                
                cm = confusion_matrix(y_true=real, y_pred=prediction)
                fig, ax = plot_confusion_matrix(conf_mat=cm,
                                                show_absolute=True,
                                                show_normed=True,
                                                colorbar=True,
                                                class_names=[0, 1])
                ax.set_title(f'{my_arguments.DATASET_NAME}_{my_arguments.PATCH_SIZE}')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                
                plt.savefig(self.args.train_savename + '/confusion_matrix.png')
                print("Saved Confusion Matrix")
            
            # torch.save(self.network.state_dict(),
            #            self.args.train_savename + '/model_weight.pkl')
                
        print('####################\tFinished Training!')
        wandb.finish()
        
        