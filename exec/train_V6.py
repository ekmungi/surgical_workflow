import sys

# sys.path.append('/home/avemuri/DEV/src/surgical_workflow/')
sys.path.append('/media/anant/dev/src/surgical_workflow/')

from dataloader.workflow_dataset import kFoldWorkflowSplit
from model.workflow_resnet_model import ResFeatureExtractor
from utils.early_stopping import EarlyStopping
from utils.helpers import ModelCheckpoint, CumulativeMovingAvgStd, ProgressBar, Engine, create_plot_window, BestScore
from utils.optimizer_utils import get_optimizer

from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch


# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, ExponentialLR

# from trixi.logger import PytorchVisdomLogger
from visdom import Visdom

import numpy as np

import os, time
from optparse import OptionParser
from tqdm import tqdm, trange

torch.backends.cudnn.benchmark = True
        

def train():

    # base_folder = "/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/COMPRESSED_0_05/TrainingSet"
    base_folder = "/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet"
    ## ======================================= Data ======================================= ##
    # image_transform = Compose([Resize(data_options['image_size'])])
    # image_transform = Compose([Resize(data_options['image_size']), ToTensor()])
    image_transform = Compose([#Resize((480,270)), 
                                ToTensor()])
                                # Normalize(mean=[0.485, 0.2131, 0.406],
                                #             std=[0.229, 0.224, 0.225])])#,
                                # Normalize(mean=[0.3610,0.2131,0.2324],
                                #             std=[0.0624,0.0463,0.0668])])
    kfoldWorkflowSet = kFoldWorkflowSplit(base_folder, 
                                            image_transform=image_transform,
                                            video_extn='.avi', shuffle=True,
                                            n_folds=21, num_phases=14,
                                            batch_size=32, num_workers=12)
    ## ======================================= Data ======================================= ##
    
    nfolds_training_loss_avg = CumulativeMovingAvgStd()
    nfolds_validation_loss_avg = CumulativeMovingAvgStd()
    nfolds_validation_score_avg = CumulativeMovingAvgStd()

    # folds_pbar = ProgressBar(kfoldWorkflowSet, desc="Folds", pb_len=1)
    # max_folds = folds_pbar.total

    epoch_msg_dict = {}

    duration = {}

    EPOCHS = 5
    ITERATIONS = 60
    FOLDS = 2

    for iFold in range(FOLDS):
        train_loader, valid_loader = next(kfoldWorkflowSet)
        train_loader = iter(train_loader)
        for epoch in range(EPOCHS):
            t0 = time.time()
            for iteration in trange(ITERATIONS):
                images, phase_annotations = next(train_loader)
            t1 = time.time()
            print('\nFold {0}, Epoch {1} : {2}\n'.format(iFold, epoch, np.round(t1-t0,2)))
        # time.sleep(0.01)
        #del(train_loader)
        #del(valid_loader)
        # if iFold+1 == FOLDS:
        #     break


    # for iFold, (train_loader, val_loader) in enumerate(folds_pbar): #= next(kfoldWorkflowSet)
        
    #     epoch_pbar = ProgressBar(range(10), desc="Epochs") #tqdm(range(epochs))
        

    #     for epoch in epoch_pbar:
            
    #         iteration_pbar = ProgressBar(train_loader, desc="Iteration", pb_len=60)
    #         max_iterations = iteration_pbar.total
    #         t0 = time.time()
    #         for iteration, (images, phase_annotations) in enumerate(iteration_pbar):

    #             # images.to(device=device), phase_annotations.to(device=device)
            
    #             # epoch_msg_dict['IMAGE'] = images.shape
    #             # epoch_msg_dict['PHASE'] = phase_annotations.shape
    #             epoch_msg_dict['EPOCH'] = epoch
    #             folds_pbar.update_message(msg_dict=epoch_msg_dict)

    #             if iteration == max_iterations:
    #                 iteration_pbar.refresh()
    #                 iteration_pbar.close()
    #                 break
    #         t1 = time.time()
    #         key = 'Fold_'+str(iFold)+'_Epoch_'+str(epoch)
    #         duration[key] = t1-t0


    #     if (iFold+1) == max_folds:
    #         folds_pbar.refresh()
    #         folds_pbar.close()
    #         break

    # print('\n\n\n')
    # for k, v in duration.items():
    #     print(k, v)

    print("\n\n\n\n=================================== DONE ===================================\n\n")


if __name__ == "__main__":
    train()