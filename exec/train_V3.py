import sys

# sys.path.append('/home/avemuri/DEV/projects/endovis2018-challenge/')
sys.path.append('/media/anant/dev/src/endovis/')

from argparse import ArgumentParser


import math
import typing as t

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision.models as models

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.handlers.early_stopping import EarlyStopping
from ignite.contrib.handlers import LinearCyclicalScheduler, CosineAnnealingScheduler, ProgressBar

from workflow.dataloader.workflow_dataset import kFoldWorkflowSplit
from workflow.model.workflow_resnet_model import ResFeatureExtractor
from workflow.utils.helpers import get_lr_scheduler, get_optimizer, get_loss
# from workflow.utils.cyclic_learning import LRSchedulerWithRestart

from visdom import Visdom

from ignite.metrics import Accuracy, Loss, RunningAverage





def create_plot_window(vis, xlabel, ylabel, title, name, tag):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title), name=name, win=tag)


def train(optimizer_options, data_options, logger_options, model_options, handlers):
    
    vis = Visdom(env=logger_options['vislogger_env'], port=logger_options['vislogger_port'])

    model = ResFeatureExtractor(pretrained_model=models.resnet50)


    image_transform = Compose([Resize(data_options['image_size']),
                                ToTensor()])
    kfoldWorkflowSet = kFoldWorkflowSplit(data_options['base_path'], 
                                            image_transform=image_transform,
                                            video_extn='.avi', shuffle=True,
                                            n_folds=data_options['n_folds'], num_phases=14,
                                            batch_size=data_options['batch_size'], 
                                            num_workers=data_options['n_threads'])



    train_loader, val_loader = next(kfoldWorkflowSet)#get_data_loaders(train_batch_size, val_batch_size)
    device = optimizer_options['device']

    if torch.cuda.is_available():
        device = 'cuda:0'
        model = model.to(device=device)

    optimizer = get_optimizer(model.parameters(), optimizer_options)
    criterion_CE = get_loss(optimizer_options) #nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion_CE, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'CE': Loss(criterion_CE)},
                                            device=device)

    ## ======================================= Visualize ======================================= ##
    train_loss_window = create_plot_window(vis, 
                                            '#Iterations', 'Loss', 
                                            'Training Loss', 'Training_Loss', 
                                            'Training_Loss')

    train_avg_loss_window = create_plot_window(vis, 
                                                '#Iterations', 'Loss', 
                                                'Training Average Loss', 
                                                'Training_Average_Loss', 
                                                'Training_Average_Loss')

    train_avg_accuracy_window = create_plot_window(vis, 
                                                    '#Iterations', 'Accuracy', 
                                                    'Training Average Accuracy', 
                                                    'Training_Average_Accuracy', 
                                                    'Training_Average_Accuracy')

    val_avg_loss_window = create_plot_window(vis, 
                                            '#Epochs', 'Loss', 
                                            'Validation Average Loss', 
                                            'Validation_Average_Loss', 
                                            'Validation_Average_Loss')


    val_avg_accuracy_window = create_plot_window(vis, 
                                                '#Epochs', 'Accuracy',
                                                'Validation Average Accuracy',
                                                'Validation_Average_Accuracy',
                                                'Validation_Average_Accuracy')
    ## ======================================= Visualize ======================================= ##


    running_average = RunningAverage(output_transform=lambda x: x)
    running_average.attach(trainer, 'loss')
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=None, output_transform=lambda x: x)
 
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % logger_options['vislogger_interval'] == 0:
            vis.line(X=np.array([engine.state.iteration]),
                     Y=np.array([engine.state.output]),
                     update='append', win=train_loss_window, name='Training_Loss')

            # pbar.log_message_pb("Loss: {:.4f}".format(engine.state.output))
            # pbar.log_message("Training Results - Epoch: {}  Avg loss: {:.2f}".format(engine.state.epoch, engine.state.output))



    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(engine):
    #     evaluator.run(train_loader)
    #     metrics = evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll = metrics['nll']
    #     pbar.log_message(
    #         "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #         .format(engine.state.epoch, avg_accuracy, avg_nll)
    #     )


    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_learning_rate(engine):
    #     iter = (engine.state.iteration - 1) % len(train_loader) + 1
    #     if iter % logger_options['vislogger_interval'] == 0:
    #         #print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
    #         #      "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
    #         lr_val = engine.state.param_history['lr'][engine.state.iteration]
    #         vis.show_value(value=np.array([lr_val]),#train_loss.item(), 
    #                             name='Learning_rate', 
    #                             label='Learning rate', 
    #                             counter=np.array([engine.state.iteration]))

    # Attach scheduler(s)
    if handlers is not None:
        for handler_args in handlers:
            # print(handler_args[0])
            (scheduler_cls, param_name, start_value, end_value, cycle_mult) = handler_args[0]
            handler = scheduler_cls(
                optimizer, param_name, start_value, end_value, len(train_loader),
                cycle_mult=cycle_mult, save_history=True)
            trainer.add_event_handler(Events.ITERATION_COMPLETED, handler)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_ce = metrics['CE']
        #print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        #      .format(engine.state.epoch, avg_accuracy, avg_ce))
        # vis.show_value(value=np.array([avg_accuracy]),#train_loss.item(), 
        #                 name='Training_Average_Accuracy', 
        #                 label='Training Average Accuracy', 
        #                 counter=np.array([engine.state.epoch]))

        # vis.show_value(value=np.array([avg_ce]),#train_loss.item(), 
        #                 name='Training_Average_Loss', 
        #                 label='Training Average Loss', 
        #                 counter=np.array([engine.state.epoch]))
        
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]),
                 win=train_avg_accuracy_window, update='append', name='Training_Average_Accuracy')
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_ce]),
                 win=train_avg_loss_window, update='append', name='Training_Average_Loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        ave_ce = metrics['CE']
        # print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            #   .format(engine.state.epoch, avg_accuracy, ave_ce))
        
        # vis.show_value(value=np.array([avg_accuracy]),#train_loss.item(), 
        #                 name='Validation_Average_Accuracy', 
        #                 label='Validation Average Accuracy', 
        #                 counter=np.array([engine.state.epoch]))

        # vis.show_value(value=np.array([avg_ce]),#train_loss.item(), 
        #                 name='Validation_Average_Loss', 
        #                 label='Validation Average Loss', 
        #                 counter=np.array([engine.state.epoch]))
        
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]),
                 win=val_avg_accuracy_window, update='append', name='Validation_Average_Accuracy')
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([ave_ce]),
                 win=val_avg_loss_window, update='append', name='Validation_Average_Loss')

    # kick everything off
    trainer.run(train_loader, max_epochs=data_options['epochs'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_file", type=str, default=None, help="log file to log output to")

    args = parser.parse_args()

    # lr=1e-3
    handlers = [ (LinearCyclicalScheduler, 'lr', args.lr, args.lr * 100, 1) ]

    train(args.batch_size, args.val_batch_size, args.epochs, args.lr, 
        args.momentum, args.log_interval, handlers)




# if __name__ == '__main__':

#     lr=1e-3
#     handlers = [ (LinearCyclicalScheduler, 'lr', lr, lr * 100, 1) ]
#     # scheduler = LinearCyclicalScheduler(optimizer, 'lr', 1, 0, 10)

#     train(base_path='/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/',
#           epochs=1, n_folds=4, lr=lr, momentum=0.95, log_interval=50, handlers=handlers)