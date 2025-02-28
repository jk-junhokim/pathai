import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import my_arguments
cudnn.benchmark = True

def MakePredictions(dataloader, model):
    real = []
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for i, (img, label, img_name) in tqdm(enumerate(dataloader)):
            
            img, label = img.cuda(), label.cuda()
            
            output = model(img)
            predicted = (output > my_arguments.THRESHOLD).long()
            
            real.extend(label.cpu().numpy().tolist())
            predictions.extend(predicted[:, 1].cpu().numpy().tolist())
            
        real = np.array(real)
        predictions = np.array(predictions)
            
    return real, predictions


def EvalMetrics(real, prediction):
    
    TP = ((real == 1) & (prediction == 1)).sum()
    FN = ((real == 1) & (prediction == 0)).sum()
    TN = ((real == 0) & (prediction == 0)).sum()
    FP = ((real == 0) & (prediction == 1)).sum()
    
    res = {}
    res['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    res['Specificity'] = TN / (TN + FP)
    res['Recall'] = TP / (TP + FN)
    res['Precision'] = TP / (TP + FP)
    res['F1-Score'] = (2 * res['Recall'] * res['Precision']) / (res['Recall'] + res['Precision'])
    
    wandb.log({"F1-Score (Validation)": res['F1-Score']})
    wandb.log({"Accuracy (Validation)": res['Accuracy']})
    
    return res