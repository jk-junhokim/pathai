####################################################
####################################################
######  THIS IS WHERE YOU MODIFY PARAMETERS   ######
####################################################
####################################################

class MyHyperparmeter:
    
    PATCH_SIZE = 128          # 128 || 224 || 256
    CALL_PRETRAINED = False      # True || False
    NETWORK = "InceptionResNet"     # VggInception || InceptionResNet || PretrainedEfficient
    OPTIMIZER = "SGD"         # Adam || Adagrad || SGD || RMSprop
    
    TEST_LABEL = "BS_128_LR_Major"
    LOG_MEMO = "Optimizer: Adagrad, Learning Rate: 0.001, \
                Train Batch Size: 256, Validation Batch Size: 256, \
                N/A, step_size: N/A, gamma: N/A"   
    EPOCH = 150 
    GPU_ID = '0'           # 0 || 1
    DATASET_NAME = "50"    # 453 || OF_Large || patch_128 || 452
    LOSS_F = "BCEWithLogitsLoss"   # CrossEntropyLoss || BCELoss || BCEWithLogitsLoss
    TRAIN_BATCH_SIZE = 256       # 16 || 32 || 64 || 128 || 256
    THRESHOLD = 0.5            # 0.5 || 0.3
    RUN_NAME = f"{NETWORK}-{OPTIMIZER}-{TEST_LABEL}"
    
    # 452 + 0.3
    # 453 + 128/256/384 + 0.5
    # 50 + 256 + 0.3
    
    # * PreprocessArgs
    base_raw_wsi_path = "/mnt/tesser_nas2/AI_DataSets/DC_Data_Storage/2023_07_06_DCGEN_453_original_train"
    base_data_path = f"/mnt/tesser_nas2/AI_DataSets/DC_Data_Storage/{DATASET_NAME}" # ! for main
    
    # * ModelArgs
    # base_data_path = same as above
    base_save_trained_models = f"../models/{NETWORK}/{DATASET_NAME}_patch_{PATCH_SIZE}_{TEST_LABEL}"

    # * WholeSlideArgs
    # test_wsi_path = "/mnt/tesser_nas2/AI_DataSets/DC_Data_Storage/2023_07_06_DCGEN_10_original_test" # ! for main
    test_wsi_path = f"/mnt/tesser_nas2/AI_DataSets/junho/Changseok/img/OncoFree/Test"
    base_save_heatmap = f"/mnt/tesser_nas2/AI_DataSets/DC_Data_Storage/{NETWORK}/{DATASET_NAME}_{PATCH_SIZE}_{TEST_LABEL}"
    

# ? Patch size = 128
# ? Optimizer = Adagrad
# ? Normalization = Yes
# ? Model = Undecided

"""
### Hyperparmeters
1. Network
2. Dataset
3. Patch Size
4. Loss Function
5. Optimizer
6. Learning Rate

models/
├── VggInception/
|       ├── 453_patch_256_Test_1/
|       |       ├── confusion_matrix.png
|       |       ├── model_info.log
|       |       ├── test_pred_score.npz
|       |       └── best_accuracy.pkl
|       |
|       ├── 453_patch_256_Test_2/
|       |       └── ...
|       |
|       ├── 453_patch_384_Test_1/
|       |       └── ...
|       └── ...
|
├── PretrainedResNet/
|       └── ...
|
├── ConvNeXt/
|       └── ...
└── ...
"""
################################################################################ 
#################################   BETA VER   #################################
################################################################################ 
# PreprocessArgs
# base_train_raw_path = "/mnt/tesser_nas2/AI_DataSets/junho/Changseok/img/OncoFree/Train"
# preprocessed_wsi_path = "/mnt/tesser_nas2/AI_DataSets/junho/Changseok/img/data"

# ModelArgs
# preprocessed_wsi_path = f"/mnt/tesser_nas2/AI_DataSets/junho/Changseok/img/data/{DATASET_NAME}_patch_{PATCH_SIZE_STR}/dataset"
# base_save_trained_models = f"../models/{NETWORK}/{DATASET_NAME}_patch_{PATCH_SIZE_STR}_results"

# WholeSlideArgs
# base_save_trained_models = f"../models/{NETWORK}/{DATASET_NAME}_patch_{PATCH_SIZE_STR}_results"
# test_wsi_path = "/mnt/tesser_nas2/AI_DataSets/junho/Changseok/img/OncoFree/Test"
# base_save_wsi_heatmap = "/mnt/tesser_nas2/AI_DataSets/DC_Data_Storage"