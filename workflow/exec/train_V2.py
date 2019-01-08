import sys

# sys.path.append('/home/avemuri/DEV/projects/endovis2018-challenge/')
sys.path.append('/media/anant/dev/src/endovis/')


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
from ignite.contrib.handlers import LinearCyclicalScheduler

from workflow.dataloader.workflow_dataset import kFoldWorkflowSplit
from workflow.model.workflow_resnet_model import ResFeatureExtractor
from workflow.utils.cyclic_learning import LRSchedulerWithRestart



def train(base_path: str,
            epochs: int,
            n_folds: int,
            #val_batch_size: int,
            lr: t.Optional[float] = 1e-2,
            momentum: t.Optional[float] = 0.5,
            log_interval: t.Optional[int] = 50,
            random_seed: t.Optional[int] = 42,
            handlers: t.Optional[t.Tuple] = ()
          ) -> nn.Module:
    """
    Instantiates and trains a CNN on MNIST.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = ResFeatureExtractor(pretrained_model=models.resnet50)


    image_transform = Compose([Resize((320,180)),
                                ToTensor()])
    kfoldWorkflowSet = kFoldWorkflowSplit(base_path, 
                                            image_transform=image_transform,
                                            video_extn='.avi', shuffle=True,
                                            n_folds=n_folds, num_phases=14,
                                            batch_size=32, num_workers=16)

    train_loader, val_loader = next(kfoldWorkflowSet)#get_data_loaders(train_batch_size, val_batch_size)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda:0'
        model = model.to(device=device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion_CE = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion_CE,
        device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': CategoricalAccuracy(), 'CE': Loss(criterion_CE)},
        device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        i = (engine.state.iteration - 1) % len(train_loader) + 1
        if i % log_interval == 0:
            print(f"[{engine.state.epoch}] {i}/{len(train_loader)} loss: {'%.2f' % engine.state.output}")

    # Attach scheduler(s)
    for handler_args in handlers:
        (scheduler_cls, param_name, start_value, end_value, cycle_mult) = handler_args
        handler = scheduler_cls(
            optimizer, param_name, start_value, end_value, len(train_loader),
            cycle_mult=cycle_mult, save_history=True)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_CE = metrics['CE']
        print("Validation Accuracy: {:.2f} Loss: {:.2f}\n".format(avg_accuracy, avg_CE))

    trainer.run(train_loader, max_epochs=epochs)
    
    return (model, trainer.state)




if __name__ == '__main__':

    lr=1e-3
    handlers = [ (LinearCyclicalScheduler, 'lr', lr, lr * 100, 1) ]
    # scheduler = LinearCyclicalScheduler(optimizer, 'lr', 1, 0, 10)

    train(base_path='/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/',
          epochs=1, n_folds=4, lr=lr, momentum=0.95, log_interval=50, handlers=handlers)