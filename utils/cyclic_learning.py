import sys

# sys.path.append('/home/avemuri/DEV/projects/endovis2018-challenge/')
sys.path.append('/media/anant/dev/src/endovis/')


import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, ExponentialLR

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from workflow.model.workflow_resnet_model import ResFeatureExtractor
from workflow.utils.helpers import create_plot_window, get_optimizer

class LRSchedulerWithRestart(_LRScheduler):
    """Proxy learning scheduler with restarts: learning rate follows input scheduler strategy but
    the strategy can restart when passed a defined number of epochs. Ideas are taken from SGDR paper.
    Args:
        scheduler (_LRScheduler): input lr scheduler
        restart_every (int): restart input lr scheduler every `restart_every` epoch.
        restart_factor (float): factor to rescale `restart_every` after each restart.
            For example, if `restart_factor=0.5` then next restart occurs in half of `restart_every` epochs.
        init_lr_factor (float): factor to rescale base lr after each restart.
            For example, if base lr of the input scheduler is 0.01 and `init_lr_factor=0.5`, then after the restart
            base lr of the input scheduler will be `0.01 * 0.5`.
    Learning rate strategy formula:
    ```
    t[-1] = 0 # Internal epoch timer dependant of global epoch value
    ...
    t[e] = t[e-1] + 1
    if t[e] % restart_every == 0:
        t[e] = 0
        restart_every *= restart_factor
        scheduler.base_lrs = scheduler.base_lrs * init_lr_factor
    scheduler.last_epoch = t[e]
    lr[e] = scheduler.get_lr()
    ```
    """

    def __init__(self, scheduler, restart_every, restart_factor=1.0, init_lr_factor=1.0, 
                 after_n_batches=1, verbose=False):
        self.scheduler = scheduler
        self.restart_every = restart_every
        self.restart_factor = restart_factor
        self.init_lr_factor = init_lr_factor
        self._t = -1
        self._t_batch = -1
        self.verbose = verbose
        self.after_n_batches = after_n_batches
        # Do not call super method as optimizer is already setup by input scheduler
        # super(LRSchedulerWithRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.scheduler.get_lr()

    def step(self, epoch=None):

        self._t += 1
        # if epoch is None:
        #     epoch = self._t + 1
        # self._t = epoch

        if self._t % self.after_n_batches == 0:
            self._t_batch += 1
            

        if self.restart_every > 0 and self.scheduler.last_epoch > 0 and \
                self._t % self.restart_every == 0:
            self._t = 0
            self._t_batch = 0
            self.restart_every = int(self.restart_every * self.restart_factor)
            self.scheduler.base_lrs = [lr * self.init_lr_factor for lr in self.scheduler.base_lrs]
            if self.verbose:
                print("LRSchedulerWithRestart: restart lr at epoch %i, next restart at %i"
                      % (self.scheduler.last_epoch, self.scheduler.last_epoch + self.restart_every))

        if self._t % self.after_n_batches == 0:
            self.scheduler.step(self._t_batch)



class LRSchedulerWithRestart_V2:
    '''
    @TODO: Choose the regular scheduler if restart is not specified
    '''
    def __init__(self, scheduler_type, n_restarts, n_lr_updates,
                    restart_factor=1.0, init_lr_factor=1.0, eta_min=None, vis=None):
        self.scheduler_type = scheduler_type
        self.eta_min = eta_min
        self.n_restarts = n_restarts
        self.n_lr_updates = n_lr_updates
        self.restart_factor = restart_factor
        self.init_lr_factor = init_lr_factor
        
        self.vis = vis


    def _process(self):
        self.restart_every = int(self.cycle_length/self.n_restarts)
        self.after_n_batches = int(self.restart_every/self.n_lr_updates)

        lr_scheduler = None
        if (self.scheduler_type == 'exponential') or (self.scheduler_type == 'exponential_R'):
            lr_scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        elif (self.scheduler_type == 'cosine_annealing_R'):
            lr_scheduler = CosineAnnealingLR(self.optimizer, 
                                                T_max=int(self.restart_every/self.after_n_batches), 
                                                eta_min=self.eta_min)
        elif (self.scheduler_type == 'reduce_on_plateau'):
            pass
        return lr_scheduler
        

    def __call__(self, optimizer, cycle_length):
        self.iteration = 0
        self.cycle_length = cycle_length
        self.optimizer = optimizer
        self.lr_scheduler = self._process()
        if (self.lr_scheduler is not None):
            if (self.scheduler_type[-2:] == '_R'):
                self.lr_scheduler = LRSchedulerWithRestart(self.lr_scheduler, 
                                                            restart_every=self.restart_every,
                                                            restart_factor=self.restart_factor,
                                                            init_lr_factor=self.init_lr_factor,
                                                            after_n_batches=self.after_n_batches)

            if self.vis is not None:
                create_plot_window(self.vis, "Iterations", "Learning rate", 
                                    'Learning rate', tag='Learning_Rate', name='Learning Rate')

    def step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        else:
            pass

        if (self.vis is not None) and (self.lr_scheduler is not None):
            self.vis.line(X=np.array([self.iteration]), 
                            Y=np.array([self.get_lr()]),
                            update='append', win='Learning_Rate', 
                            name='Learning rate')

        self.iteration += 1


    def get_lr(self):
        if self.lr_scheduler is None:
            return None
        else:
            return self.lr_scheduler.get_lr()[0]



def test(schedular_type):

    print("\n\n")
    EPOCHS = 1
    ITERATIONS = 15#000
    N_RESTARTS = 1
    N_LR_UPDATES = 5
    RESTART_FACTOR = 1.
    INIT_LR_FACTOR = 1.
    ETA_MIN = 1e-6
    CYCLE_LENGTH = 7
    LR = 1e-4

    # optimizer = get_optimizer(optimizer_val, learning_rate, model.parameters())
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    model = ResFeatureExtractor()

    optimizer_ops = {'optimizer': 'adam',
                        'learning_rate': LR}


    optimizer = get_optimizer(model.parameters(), optimizer_ops)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

    scheduler = LRSchedulerWithRestart_V2(scheduler_type=schedular_type, 
                                            n_restarts=N_RESTARTS, 
                                            n_lr_updates=N_LR_UPDATES,
                                            restart_factor=RESTART_FACTOR, 
                                            init_lr_factor=INIT_LR_FACTOR,
                                            eta_min=ETA_MIN)
    scheduler(optimizer, CYCLE_LENGTH)

    # lr_scheduler_restarts = get_lr_scheduler(schedular_type=schedular_type, 
    #                                             optimizer=optimizer, 
    #                                             max_iterations=ITERATIONS, 
    #                                             n_restarts=N_RESTARTS, n_lr_updates=N_LR_UPDATES,
    #                                             restart_factor=1.0, init_lr_factor=0.9)

    learning_rate = []
    x_list = []

    epoch_pbar = tqdm(range(EPOCHS))
    for epoch in epoch_pbar:
        iteration_pbar = tqdm(range(ITERATIONS))
        for iteration in iteration_pbar:

            scheduler.step()
            
            learning_rate.append(scheduler.get_lr())
            x_list.append(iteration)

            print(optimizer.param_groups[0]['lr'])

            # count = 0
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])
            #     count += 1

            # print(count)



    print("\n\n")
    sns.lineplot(x_list, learning_rate)#, style='s')
    plt.grid()
    plt.show()

    

if __name__ == "__main__":
    test(schedular_type='cosine_annealing_R')