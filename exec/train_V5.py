import sys

# sys.path.append('/home/avemuri/DEV/src/surgical_workflow/')
# sys.path.append('/media/anant/dev/src/surgical_workflow/')

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

import os
from optparse import OptionParser
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def predict(evaluator, max_iterations=None, device='cpu', vis=None):
        if vis is not None:
            create_plot_window(vis, "Iterations", "CE Loss", "Validation loss", 
                                tag='Validation_Loss', name='Validation Loss')
            create_plot_window(vis, "Iterations", "Score", "Validation Score", 
                                tag='Validation_Score', name='Validation Score')
        
        avg_loss = CumulativeMovingAvgStd()
        avg_score = CumulativeMovingAvgStd()

        pbar = ProgressBar(evaluator.data_iterator, desc="Predict", pb_len=max_iterations)
        max_iterations = pbar.total

        msg_dict = {}
        for iteration, (x, y) in enumerate(pbar):
            # print(x.shape, y.shape)
            loss, score = evaluator(x.to(device), y.to(device))
            avg_loss.update(loss)
            avg_score.update(score)
            msg_dict['APL'] = avg_loss.get_value()[0]
            msg_dict['APS'] = avg_score.get_value()[0]

            if vis is not None:
                vis.line(X=np.array([iteration/pbar.total]), 
                                    Y=np.array([avg_loss.get_value()[0]]),
                                    update='append', win='Validation_Loss', 
                                    name='Validation Loss')
                vis.line(X=np.array([iteration/pbar.total]), 
                                    Y=np.array([avg_score.get_value()[0]]),
                                    update='append', win='Validation_Score', 
                                    name='Validation Score')

            pbar.update_message(msg_dict=msg_dict)
            
            if iteration == max_iterations:
                pbar.close()
                break


        return avg_loss.get_value()[0], avg_score.get_value()[0]


def runEpoch(loader, model, loss_fn, optimizer, scheduler, device, 
                vis, epoch, iFold, folds_pbar, avg_training_loss,
                avg_training_score, logger_options, optimizer_options, 
                msg_dict):

    ## ======================================= Early Stop ======================================= ##
    early_stop = False
    if not (optimizer_options['early_stopping'] == ""):
        #['min', '0.01', '21']
        mode = optimizer_options['early_stopping'][0]
        min_delta = float(optimizer_options['early_stopping'][1])
        patience = int(optimizer_options['early_stopping'][2])
        early_stopping = EarlyStopping(mode=mode, min_delta=min_delta, patience=patience)
    ## ======================================= Early Stop ======================================= ##
    
    
    trainer = Engine(model, optimizer, loss_fn, scheduler, loader, 
                            optimizer_options["accumulate_count"], device,
                            use_half_precision=optimizer_options["use_half_precision"],
                            score_type="f1")
    

    iteration_pbar = ProgressBar(loader, desc="Iteration", pb_len=optimizer_options['max_iterations'])
    max_iterations = iteration_pbar.total

    for iteration, (images, phase_annotations) in enumerate(iteration_pbar):

        ### ============================== Training ============================== ###
        train_loss, train_score = trainer(images.to(device=device), phase_annotations.to(device=device))
        avg_training_loss.update(train_loss)
        avg_training_score.update(train_score)
        msg_dict['ATL'] = avg_training_loss.get_value()[0]
        msg_dict['ATS'] = avg_training_score.get_value()[0]
        ### ============================== Training ============================== ###
        
        ### ============================== Plot ============================== ###
        if ((iteration) % logger_options["vislogger_interval"] == 0):
            # print(avg_training_loss.get_value()[0])
            vis.line(X=np.array([epoch + (iteration/iteration_pbar.total)]), 
                        Y=np.array([avg_training_loss.get_value()[0]]),
                        update='append', win='Training_Loss_Fold_'+str(iFold+1), 
                        name='Training Loss Fold '+str(iFold+1))
        ### ============================== Plot ============================== ###

        if early_stop:
            iteration_pbar.close()
            print("\n==========================\nEarly stop\n==========================\n")
            break                
        
        folds_pbar.update_message(msg_dict=msg_dict)
        
        if iteration == max_iterations:
            iteration_pbar.refresh()
            iteration_pbar.close()
            break


        

def train(optimizer_options, data_options, logger_options, model_options, scheduler_options):#, results_path=None):
    
    #torch.manual_seed(42)
    #np.random.seed(42)

    vis = Visdom(env=logger_options['vislogger_env'], port=logger_options['vislogger_port'])    
    device = torch.device(optimizer_options['device'])
    epochs = optimizer_options['epochs']
    
    

    ## ======================================= Scheduler ======================================= ##
    
    ## ======================================= Scheduler ======================================= ##

    ## ======================================= Save model ======================================= ##
    if (logger_options['save_model'] == ""):
        model_checkpoint = ModelCheckpoint()
    else:
        suffix = optimizer_options['optimizer']+"_"+str(optimizer_options['learning_rate'])+"_"+logger_options['suffix']
        model_checkpoint = ModelCheckpoint(save_model=True, save_path=logger_options['save_model'], 
                                            use_loss=True, suffix=suffix) ####### FILL PARAMTERS!!!!
    ## ======================================= Save model ======================================= ##

    ## ======================================= Data ======================================= ##
    # image_transform = Compose([Resize(data_options['image_size'])])
    # image_transform = Compose([Resize(data_options['image_size']), ToTensor()])
    image_transform = Compose([#Resize(data_options['image_size']), 
                                ToTensor()
                                ])
                                # Normalize(mean=[0.485, 0.2131, 0.406],
                                #             std=[0.229, 0.224, 0.225])])#,
                                # Normalize(mean=[0.3610,0.2131,0.2324],
                                #             std=[0.0624,0.0463,0.0668])])
    kfoldWorkflowSet = kFoldWorkflowSplit(data_options['base_path'], 
                                            image_transform=image_transform,
                                            video_extn='.avi', shuffle=True,
                                            n_folds=data_options['n_folds'], num_phases=14,
                                            batch_size=data_options['batch_size'], 
                                            num_workers=data_options['n_threads'],
                                            video_folder='videos_480x272')
    ## ======================================= Data ======================================= ##
    
    nfolds_training_loss_avg = CumulativeMovingAvgStd()
    nfolds_validation_loss_avg = CumulativeMovingAvgStd()
    nfolds_validation_score_avg = CumulativeMovingAvgStd()

    folds_pbar = ProgressBar(kfoldWorkflowSet, desc="Folds", pb_len=optimizer_options['run_nfolds'])
    max_folds = folds_pbar.total


    for iFold, (train_loader, val_loader) in enumerate(folds_pbar): #= next(kfoldWorkflowSet)

        ## ======================================= Create Plot ======================================= ##
        create_plot_window(vis, "Epochs+Iterations", "CE Loss", "Training loss Fold "+str(iFold+1), 
                            tag='Training_Loss_Fold_'+str(iFold+1), name='Training Loss Fold '+str(iFold+1))
        create_plot_window(vis, "Epochs+Iterations", "CE Loss", "Validation loss Fold "+str(iFold+1), 
                            tag='Validation_Loss_Fold_'+str(iFold+1), name='Validation Loss Fold '+str(iFold+1))
        create_plot_window(vis, "Epochs+Iterations", "Score", "Validation Score Fold "+str(iFold+1), 
                            tag='Validation_Score_Fold_'+str(iFold+1), name='Validation Loss Fold '+str(iFold+1))
        ## ======================================= Create Plot ======================================= ##


        ## ======================================= Model ======================================= ##
        # TODO: Pass 'models.resnet50' as string

        model = ResFeatureExtractor(pretrained_model=models.resnet101, 
                                        device=device)
                                        
        if model_options['pretrained'] is not None:
            # print('Loading pretrained model...') 
            checkpoint = torch.load(model_options['pretrained'])
            model.load_state_dict(checkpoint['model'])

        
        ## ======================================= Model ======================================= ##


        ### ============================== Parts of Training step ============================== ###
        criterion_CE = nn.CrossEntropyLoss().to(device)        
        ### ============================== Parts of Training step ============================== ###
        
        epoch_pbar = ProgressBar(range(epochs), desc="Epochs") #tqdm(range(epochs))
        epoch_training_avg_loss = CumulativeMovingAvgStd()
        epoch_training_avg_score = CumulativeMovingAvgStd()
        epoch_validation_loss = BestScore()
        # epoch_validation_score = BestScore()
        epoch_msg_dict = {}

        evaluator = Engine(model, None, criterion_CE, None, val_loader, 0, device, False,
                            use_half_precision=optimizer_options["use_half_precision"],
                            score_type="f1")

        for epoch in epoch_pbar:

            if (optimizer_options['switch_optimizer'] > 0) and ((epoch+1) % optimizer_options['switch_optimizer'] == 0):
                temp_optimizer_options = optimizer_options
                temp_optimizer_options['optimizer'] = 'sgd'
                temp_optimizer_options['learning_rate'] = 1e-3
                optimizer, scheduler = get_optimizer(model.parameters(), temp_optimizer_options, scheduler_options, train_loader, vis)
            else:
                optimizer, scheduler = get_optimizer(model.parameters(), optimizer_options, scheduler_options, train_loader, vis)
            # else:
            #     optimizer, scheduler = get_optimizer(model.parameters(), optimizer_options, scheduler_options, train_loader, vis)

            runEpoch(train_loader, model, criterion_CE, optimizer, scheduler, device, 
                        vis, epoch, iFold, folds_pbar, epoch_training_avg_loss,
                        epoch_training_avg_score, logger_options, optimizer_options, 
                        epoch_msg_dict)

            ### ============================== Validation ============================== ###
            validation_loss, validation_score = None, None
            if (optimizer_options["validation_interval_epochs"] > 0):
                if ((epoch+1) % optimizer_options["validation_interval_epochs"] == 0):
                    validation_loss, validation_score = predict(evaluator, 
                                                                optimizer_options['max_valid_iterations'], 
                                                                device, vis)
                    epoch_validation_loss.step(validation_loss, [validation_score])
                    # epoch_validation_score.step(validation_score)
                    vis.line(X=np.array([epoch]), 
                                Y=np.array([validation_loss]),
                                update='append', win='Validation_Loss_Fold_'+str(iFold+1), 
                                name='Validation Loss Fold '+str(iFold+1))
                    vis.line(X=np.array([epoch]), 
                                Y=np.array([validation_score]),
                                update='append', win='Validation_Score_Fold_'+str(iFold+1), 
                                name='Validation Score Fold '+str(iFold+1))
                    epoch_msg_dict['CVL'] = validation_loss
                    epoch_msg_dict['CVS'] = validation_score
                    epoch_msg_dict['BVL'] = epoch_validation_loss.score()[0]
                    epoch_msg_dict['BVS'] = epoch_validation_loss.score()[1][0]
                    folds_pbar.update_message(msg_dict=epoch_msg_dict)
            ### ============================== Validation ============================== ###


            ### ============================== Save model ============================== ###
            model_checkpoint.step(curr_loss=validation_loss,
                                    model=model, suffix='_Fold_'+str(iFold))
            vis.save([logger_options['vislogger_env']])
            ### ============================== Save model ============================== ###
            
            # if early_stop:
            #     epoch_pbar.close()
            #     break

            # torch.cuda.empty_cache()

        if (iFold+1) == max_folds:
            folds_pbar.refresh()
            folds_pbar.close()
            break

    print("\n\n\n\n=================================== DONE ===================================\n\n")

    

def test(base_path, pretrained_model=None):

    if pretrained_model is not None:
        checkpoint = torch.load(pretrained_model)
        model = checkpoint['model']

        if torch.cuda.is_available():
            model.cuda()

        criterion_CE = nn.CrossEntropyLoss()

        imgaug_transform, test_loader = get_dataloaders(base_path, shuffle=False)

        loss, acc = model.predict(test_loader, criterion_CE, print_results=True)

        print('Loss: {0}, Accuracy: {1}'.format(loss, acc))
