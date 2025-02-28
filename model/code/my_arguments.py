from __future__ import print_function
import os
import torch
import argparse
import logging
from s1_param import MyHyperparmeter as hp


class ModelArgs:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Classification CNN Trainer')
        
        # * General Parameters
        default_data_path = f"{hp.base_data_path}/patch_{hp.PATCH_SIZE}/dataset"
        parser.add_argument('-G', '--gpu-ids', default=hp.GPU_ID, type=str, required=False, help='one or multi gpus. gpu ids: e.g.0, 1. use -1 for CPU')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('-W', '--nWorker', default=16, type=int, required=False, help='Num worker for dataloader, default is 16.')
        parser.add_argument('-C', '--nCls', default=2, type=int, required=False, help='num of Class, here is 2.')
        parser.add_argument('-S', '--seed', default=1, type=int, required=False, help='random seed, default 1.')
        
        # * Patch-wise Training
        parser.add_argument('-trainpath', '--trainpath', default=f'{default_data_path}/train.txt', type=str, required=False, help='trainpath, default is train.txt.')
        parser.add_argument('-validpath', '--validpath', default=f'{default_data_path}/val.txt', type=str, required=False, help='valpath, default is val.txt.')
        parser.add_argument('-network', '--network', default=hp.NETWORK, type=str, required=False, help='hp.NETWORK vgg + inception w/ bn, default is VggInception')
        parser.add_argument('-loss', '--loss', default=hp.LOSS_F, type=str, required=False, help='loss function for classification, default is BCEWithLogitsLoss')
        parser.add_argument('-optim', '--optimizer', default=hp.OPTIMIZER, type=str, required=False, help='optimizer for classification, default is SGD')
        parser.add_argument('-E', '--epoches', default=hp.EPOCH, type=int, required=False, help='hp.EPOCH, default is 150.')
        parser.add_argument('-TB', '--train-batch-size', default=hp.TRAIN_BATCH_SIZE, type=int, required=False, help='train batch size, default is 256.')
        parser.add_argument('-VB', '--val-batch-size', default=256, type=int, required=False, help='validation batch size, default is 256.')
        parser.add_argument('-ps', '--patch-size', default=str(hp.PATCH_SIZE), type=str, required=False, help='patch, default is 128.')
        
        # * Patch-wise Testing
        parser.add_argument('-testpath', '--testpath', default=f'{default_data_path}/test.txt', type=str, required=False, help='testpath, default is test.txt.')
        parser.add_argument('-TestB', '--test-batch-size', default=16, type=int, required=False, help='batch size, default is 16.')
        parser.add_argument('-R', '--is-pretrained', default=hp.CALL_PRETRAINED, type=bool, required=False, help='set to True for testing')
        parser.add_argument('-restore', '--test-restore', default=f'{hp.base_save_trained_models}/best_accuracy.pkl', type=str, required=False, help='Model path restoring for testing, if none, just \'\'.')
        
        # * Save Results
        parser.add_argument('-sw', '--train-savename', default=f'{hp.base_save_trained_models}', type=str, required=False, help='savename for model saving, default is.')
        parser.add_argument('-st', '--test-savename', default=f'{hp.base_save_trained_models}', type=str, required=False, help='savename for model saving, default is.')
        
        # * Optimizer & Learning Rate
        parser.add_argument('-Wg', '--weights', default=None, type=list, required=False, help='weights for CEloss.')
        parser.add_argument('-LR', '--initLR', default=0.001, type=float, required=False, help='init lr, default is 0.001.')
        parser.add_argument('--beta1', type=float, default=0.9, metavar='M', help='Adam beta1 (default: 0.9)')
        parser.add_argument('--beta2', type=float, default=0.999, metavar='M', help='Adam beta2 (default: 0.999)')
        parser.add_argument('-mo', '--momentum', default=0.9, type=float, required=False, help='momentum, default is 0.8.')
        parser.add_argument('-de', '--decay', default=0.001, type=float, required=False, help='decay, default is 1e-5.')
        
        self._parser = parser
        
    def parse(self):
        args = self._parser.parse_args()
        
        if args.hp.GPU_IDs != "-1":
            os.environ['CUDA_VISIBLE_DEVICES'] = args.hp.GPU_IDs
            args.cuda = not args.no_cuda and torch.cuda.is_available()
        else:
            args.cuda = False
        
        log_file_path = args.train_savename
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
        log_file = os.path.join(log_file_path, "model_info.log")
        logging.basicConfig(filename=f'{log_file}', level=logging.INFO, format='%(message)s')    
        device = torch.cuda.get_device_name(torch.cuda.current_device())
        arguments = vars(args)
        print_list = ["network", "loss", "optimizer", "epoches", "train_batch_size", "train_savename"]
        print(f"GPU Device: {device}")
        print(f"GPU ID: {hp.GPU_ID}")
        print(f"Dataset: {hp.DATASET_NAME}")
        print(f"Patch Size: {hp.PATCH_SIZE}")
        print('------------ Arguments -------------')
        logging.info(f"\n\n\nGPU Device: {device}")
        logging.info(f"GPU ID: {hp.GPU_ID}")
        logging.info(f"Dataset: {hp.DATASET_NAME}")
        logging.info(f"Patch Size: {hp.PATCH_SIZE}")
        logging.info('------------ Arguments -------------')
        for k, v in sorted(arguments.items()):
            if k in print_list:
                print('%s: %s' % (str(k), str(v)))
                logging.info('%s: %s' % (str(k), str(v)))
        logging.info(f"Pretrained: {hp.CALL_PRETRAINED}\n")
        logging.info(f"Memo: {hp.LOG_MEMO}\n")
        print(f"Pretrained: {hp.CALL_PRETRAINED}")
        print('-------------- All Set ----------------')
        return args
        
        
class PreprocessArgs:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Preprocess Whole Slide Image')
        
        parser.add_argument('-originpath', '--origin-path', default=f"{hp.base_raw_wsi_path}", type=str, required=False, help='path to whole slide images')
        parser.add_argument('-ps', '--patch-size', default=hp.PATCH_SIZE, type=int, required=False, help='patch, default is 256.')
        parser.add_argument('-ml', '--mask-level', default=0, type=int, required=False, help='mask level, default is 0.')
        parser.add_argument('-map', '--map-level', default=0, type=int, required=False, help='map level, default is 0.')
        parser.add_argument('-nt', '--normal-threshold', default=hp.THRESHOLD, type=int, required=False, help='normal mask inclusion ratio')
        # parser.add_argument('-nsm', '--normal-sel-max', default=5000, type=int, required=False, help='number limit of normal patches')
        parser.add_argument('-tt', '--tumor-threshold', default=hp.THRESHOLD, type=int, required=False, help='tumor mask inclusion ratio')
        # parser.add_argument('-tsm', '--tumor-sel-max', default=10000, type=int, required=False, help='number limit of tumor patches')
        parser.add_argument('-xmlpath', '--xml-path', default=f"{hp.base_raw_wsi_path}/annotation", type=str, required=False, help='path to GET xml annotations')
        parser.add_argument('-slidepath', '--slide-path', default=f"{hp.base_raw_wsi_path}/slide", type=str, required=False, help='path to GET whole slide image')
        parser.add_argument('-maskpath', '--mask-path', default=f"{hp.base_data_path}/patch_{hp.PATCH_SIZE}/mask", type=str, required=False, help='path to SAVE mask')
        parser.add_argument('-patchpath', '--patch-path', default=f"{hp.base_data_path}/patch_{hp.PATCH_SIZE}/patch", type=str, required=False, help='path to SAVE patch')
        parser.add_argument('-datasetpath', '--dataset-path', default=f"{hp.base_data_path}/patch_{hp.PATCH_SIZE}/dataset", type=str, required=False, help='path to SAVE final preprocessed info')
        parser.add_argument('-swa', '--width-tracker', default=0, type=int, required=False, help='average width of slide')
        parser.add_argument('-sha', '--height-tracker', default=0, type=int, required=False, help='average height of slide')
        parser.add_argument('-tracker', '--slide-tracker', default=0, type=int, required=False, help='average height of slide')
        
        self._parser = parser
    
    def parse(self):
        args = self._parser.parse_args()
        return args        
    

class WholeSlideArgs:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Breast Cancer WSI Classification')
        
        # * for wsi preprocessing
        parser.add_argument('-slidepath', '--slide-path', default=f'{hp.test_wsi_path}', type=str, required=False, help='whole slide image file path.')
        parser.add_argument('-ps', '--patch-size', default=hp.PATCH_SIZE, type=str, required=False, help='patch, default is 256.')
        parser.add_argument('-hp.NETWORK', '--hp.NETWORK', default=hp.NETWORK, type=str, required=False, help='default is VggInception')
        parser.add_argument('-masklevel', '--mask-level', default=0, type=int, required=False, help='default is 0')
        parser.add_argument('-tissuethres', '--tissue-thres', default=0.4, type=int, required=False, help='default is 0.4')
        
        # * for wsi inference
        parser.add_argument('-B', '--wsi-batch-size', default=8, type=int, required=False, help='batch size, default is 16.')
        parser.add_argument('-R', '--is-pretrained', default=hp.CALL_PRETRAINED, type=bool, required=False, help='set to True for testing')
        parser.add_argument('-restore', '--test-restore', default=f'{hp.base_save_trained_models}/best_accuracy.pkl', 
                            type=str, required=False, help='Model path restoring for testing, if none, just \'\', no default.')
        # * wsi saver
        parser.add_argument('-st', '--heatmap-saver', default=f'{hp.base_save_heatmap}', type=str, required=False,
                            help='path to save heatmap')
        # * for general
        parser.add_argument('-G', '--gpu-ids', default=hp.GPU_ID, type=str, required=False, help='one or multi gpus. gpu ids: e.g.0, 1. use -1 for CPU')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('-W', '--nWorker', default=8, type=int, required=False, help='Num worker for dataloader, default is 8.')
        parser.add_argument('-C', '--nCls', default=2, type=int, required=False, help='num of Class, here is 2.')
        parser.add_argument('-S', '--seed', default=1, type=int, required=False, help='random seed, default 1.')
    
        self._parser = parser
        
        
    def parse(self):
        args = self._parser.parse_args()
        
        if args.hp.GPU_IDs != "-1":
            os.environ['CUDA_VISIBLE_DEVICES'] = args.hp.GPU_IDs
            args.cuda = not args.no_cuda and torch.cuda.is_available()
        else:
            args.cuda = False
        
        device = torch.cuda.get_device_name(torch.cuda.current_device())
        print('\n------------ Arguments -------------')
        print(f"GPU Device: {device}")
        print(f"GPU ID: {hp.GPU_ID}")
        print(f"hp.NETWORK: {hp.NETWORK}")
        print(f"Pretrained: {hp.CALL_PRETRAINED}")
        print(f"Trained On: {hp.DATASET_NAME}_{hp.PATCH_SIZE}")
        print('-------------- All Set ----------------')
        return args