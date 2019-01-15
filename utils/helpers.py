import os
from sklearn.model_selection import KFold

from glob import glob
import numpy as np
from tqdm import tqdm
from math import sqrt
from sklearn import metrics


# from utils.cyclic_learning import LRSchedulerWithRestart
from optim.adamw import AdamW

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, ExponentialLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from ignite.contrib.handlers import LinearCyclicalScheduler, CosineAnnealingScheduler
from apex.fp16_utils import FP16_Optimizer
from apex import amp

import threading

### ============================================================================================================== ###

def fast_collate_nvidia_apex(batch):

    # print("\n\nIN COLLATE FUNCTION\n\n")
    # print(batch[1])
    imgs = [img[0] for img in batch]
    targets = [target[1] for target in batch]
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    targets = torch.from_numpy(np.array(targets))
    return tensor, targets

### ============================================================================================================== ###


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.lock = threading.Lock()
        # self.mean = torch.tensor([0.485 * 255, 0.2131 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.mean = torch.tensor([0.3610 * 255, 0.2131 * 255, 0.2324 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.0624 * 255, 0.0463 * 255, 0.0668 * 255]).cuda().view(1,3,1,1)
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()



    def preload(self):

        
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        
        self.lock.acquire()
        input = self.next_input
        target = self.next_target
        self.preload()
        self.lock.release()
        
        return input, target

    def __len__(self):
        return len(self.loader)


### ============================================================================================================== ###


def save_checkpoint(state, is_best, filename='checkpoint', suffix=""):
    if is_best:
        torch.save(state, os.path.join(filename, suffix+'_model_best.pth.tar'))
    else:
        torch.save(state, filename+'.pth.tar')

### ============================================================================================================== ###

def get_optimizer(model_parameters, optimizer_ops, scheduler_options, data_iterator):
    if optimizer_ops['optimizer']=="sgd":
        optimizer = torch.optim.SGD(model_parameters, lr=optimizer_ops['learning_rate'], 
                                    momentum=optimizer_ops['momentum'], 
                                    weight_decay=optimizer_ops['weight_decay'])
    elif optimizer_ops['optimizer']=="adam":
        optimizer = torch.optim.Adam(model_parameters, lr=optimizer_ops['learning_rate'], amsgrad=optimizer_ops['amsgrad'])
    elif optimizer_ops['optimizer']=="adamw":
        optimizer = AdamW(model_parameters, lr=optimizer_ops['learning_rate'], 
                            weight_decay=optimizer_ops['weight_decay'])


    if scheduler_options['cycle_length'] > 0:
        cycle_length = scheduler_options['cycle_length']
    elif (optimizer_options['max_iterations'] is None) or (optimizer_options['max_iterations'] <= 0):
        cycle_length = len(data_iterator)
    else:
        cycle_length = optimizer_options['max_iterations']
    
    scheduler(optimizer, cycle_length)

    # if (optimizer_ops['use_half_precision']):
    #     optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0)

    return optimizer, scheduler

### ============================================================================================================== ###

def get_loss(optimizer_ops):
    if optimizer_ops['loss_fn']=="ce":
        loss_fn = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.SGD(model_parameters, lr=optimizer_ops['learning_rate'], 
    #                                 momentum=optimizer_ops['momentum'], 
    #                                 weight_decay=optimizer_ops['weight_decay'])
    # elif optimizer_ops['optimizer']=="adam":
    #     optimizer = torch.optim.Adam(model_parameters, lr=optimizer_ops['learning_rate'])
    # elif optimizer_ops['optimizer']=="adamw":
    #     optimizer = AdamW(model_parameters, lr=optimizer_ops['learning_rate'], 
    #                         weight_decay=optimizer_ops['weight_decay'])

    return loss_fn

### ============================================================================================================== ###

def create_plot_window(vis, xlabel, ylabel, title, name, tag):
    return vis.line(X=np.array([0]), Y=np.array([np.nan]),
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title), 
                    name=name, win=tag)

### ============================================================================================================== ###
                    
class ModelCheckpoint(object):
    def __init__(self, save_model=False, save_path=None, use_loss=True, suffix=""):
        self.best_loss = 100
        self.best_accuracy = 0.0001
        self.save_model = save_model
        self.use_loss = use_loss

        if save_path is not None:
            save_path = os.path.join(save_path, 'models', suffix)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print('Model save path: {0}'.format(save_path))
        self.save_path = save_path
    
    def step(self, curr_loss=100, curr_accuracy=0.0001, model=None, suffix=""):
        if self.use_loss:
            if curr_loss is not None:
                if curr_loss < self.best_loss:
                    self.best_loss = curr_loss
                    self.best_accuracy = curr_accuracy
                    if self.save_model:
                        save_checkpoint({'model': model.state_dict()}, True, 
                                        self.save_path, suffix)

        elif curr_accuracy is not None:
            if curr_accuracy > self.best_accuracy:
                self.best_loss = curr_loss
                self.best_accuracy = curr_accuracy
                if self.save_model:
                    save_checkpoint({'model': model.state_dict()}, True, 
                                    self.save_path, suffix)


### ============================================================================================================== ###

class RunningAverage(object):
    def __init__(self, alpha=0.9):
        self._value = None
        self.alpha = alpha

    def update(self, update_val):
        if self._value is None:
            self._value = update_val
        else:
            self._value = self._value * self.alpha + (1.0 - self.alpha) * update_val

        #return self._value


    def get_value(self):
        return self._value


### ============================================================================================================== ###

class Score:
    def __init__(self, score_type='accuracy'):
        self.score_type = score_type
        if score_type == 'accuracy':
            self.score_fn = metrics.accuracy_score
        elif score_type == 'f1':
            self.score_fn = metrics.f1_score
        elif score_type == 'precision':
            self.score_fn = metrics.precision_score
        elif score_type == 'recall':
            self.score_fn = metrics.recall_score
        elif score_type == 'auc':
            self.score_fn = metrics.roc_auc_score
        elif score_type == 'jaccard':
            self.score_fn = metrics.jaccard_similarity_score

        if score_type == 'jaccard':
            self.kwargs = {'normalize': True}
        else:
            self.kwargs = {'average': 'micro'}

    def __call__(self, y_true, y_pred):
        return self.score_fn(y_true, y_pred, **self.kwargs)




### ============================================================================================================== ###

class CumulativeMovingAvgStd(object):
    def __init__(self):
        self._mean = None
        self._std = None
        self.iteration = 0

    def update(self, update_val):
        self.iteration += 1
        if self._mean is None:
            self._mean = update_val
            self._std = 0
        else:
            
            prev_mean = self._mean
            self._mean = self._mean + ((update_val-self._mean)/self.iteration)
            self._std += (update_val-self._mean)*(update_val-prev_mean)
            # self._value = (self._value * (self.iteration) + update_val)/(self.iteration+1)

        
        #return self._mean, self._std


    def get_value(self):
        if (self._std is None):
            return self._mean, None
        else:
            return self._mean, sqrt(self._std/self.iteration)

### ============================================================================================================== ###

class BestScore(object):
    def __init__(self, mode='min'):
        self._score = None
        self._additional_score_list = None
        self._init_is_better(mode)

    def step(self, val, additional_val_list=[]):
        if self._score is None:
            self._score = val
            self._additional_score_list = additional_val_list
        elif self.is_better(val, self._score):
            self._score = val
            self._additional_score = additional_val_list


    def _init_is_better(self, mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best
        if mode == 'max':
            self.is_better = lambda a, best: a > best


    def score(self):
        return self._score, self._additional_score

### ============================================================================================================== ###

def create_handlers(handler_list, handler_options, optimizer_options, 
                    model_options):
    
    handlers = []
    for handler in handler_list:
        if handler=='LinearCyclicalScheduler':
            lr = optimizer_options['learning_rate']
            handlers.append([(LinearCyclicalScheduler, 'lr', lr , 
                            lr * handler_options['cycle_length'],
                            handler_options['step_size'])])

    return handlers

### ============================================================================================================== ###

def kfold_split(video_base_path, phase_base_path, instrument_base_path, video_extn='.avi',
                n_splits=4, shuffle=False):

    phase_list_paths = glob(os.path.join(phase_base_path, '*.csv'))
    phase_list = []
    video_list = []
    instrument_list = []
    for phase_path in phase_list_paths:
        _, file_name = os.path.split(phase_path)
        file_name, _ = os.path.splitext(file_name)
        video_path = os.path.join(video_base_path, file_name+video_extn)
        if instrument_base_path is not None:
            instrument_path = os.path.join(instrument_base_path, file_name+".csv")
        if os.path.isfile(video_path) & os.path.isfile(phase_path):
            phase_list.append(phase_path)
            video_list.append(video_path)
            if instrument_base_path is not None:
                instrument_list.append(instrument_path)

    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    # kf.get_n_splits(video_list) 

    return kf, np.array(video_list), np.array(phase_list), np.array(instrument_list)

### ============================================================================================================== ###

class Engine(object):
    def __init__(self, model, optimizer, loss_fn, lr_scheduler, data_iterator=None,
                    accumulation_count=1, device='cpu', train=True, verbose=False,
                    use_half_precision=False, score_type='accuracy'):
        self.device = device

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_iterator = data_iterator
        self.accumulation_count = accumulation_count
        self.count = 0
        self.lr_scheduler = lr_scheduler
        self.train = train
        self.iteration = 0
        self.verbose = verbose
        self.use_half_precision = use_half_precision

        if self.use_half_precision:
            self.amp_handle = amp.init(enabled=True)
            # self.model = self.model.half()
        else:
            self.amp_handle = amp.init(enabled=False)

        self.score = Score(score_type=score_type)

        


    def __call__(self, x=None, y=None):


        if self.train:
            self.model.eval()
        else:
            self.model.train()

        # if self.use_half_precision:
        #     x = x.half()
        #     y = y.half()

        if self.train:
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            y_pred = self.model(x)

            loss = self.loss_fn(y_pred, y.long())
            
            self.count += 1
            if self.count == self.accumulation_count:
                # if self.use_half_precision:
                #     self.optimizer.backward(loss)
                # else:
                #     loss.backward()
                with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer.step()
                self.count = 0

            self.iteration += 1
            with torch.no_grad():
                y_pred_softmax = F.softmax(y_pred, dim=1)
                y_pred_softmax_classes = torch.max(y_pred_softmax, 1)[1].cpu().numpy()
                score = self.score(y.cpu().numpy().transpose(), y_pred_softmax_classes)
                # accuracy = np.mean(y_pred_softmax_classes == y.cpu().numpy().transpose())

                if self.verbose:
                    print('Loss: {0}, Score: {1}, GT phases: {2}, Pred: {3}'.format(np.round(loss, 3), 
                                                                                    np.round(score, 3), 
                                                                                    y.cpu().numpy().transpose(), 
                                                                                    y_pred_softmax_classes))
                return loss.item(), score

            

            
        else:
            self.iteration += 1
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)

                y_pred_softmax = F.softmax(y_pred, dim=1)
                y_pred_softmax_classes = torch.max(y_pred_softmax, 1)[1].cpu().numpy()
                score = self.score(y.cpu().numpy().transpose(), y_pred_softmax_classes)
                # accuracy = np.mean(y_pred_softmax_classes == y.cpu().numpy().transpose())

                return loss.item(), score
        
        



        # self.predictions = model(images)

        # if torch.cuda.is_available():
        #     train_loss = criterion_CE(predictions, phase_annotations.squeeze().long())
        # else:
        #     train_loss = criterion_CE(predictions, phase_annotations.squeeze().long())

        # train_loss.backward()
        # optimizer.step()


### ============================================================================================================== ###

class ProgressBar(tqdm):

    '''
    TODO: Add a member function to update the iteration length and send a break signal when its complete.
    '''
    def __init__(self, iterator, desc, pb_len=None, device='cpu'):
        
        if (pb_len < 1) and (pb_len > 0):
            pb_len = int(len(iterator)*pb_len)
        elif (pb_len is None) or (pb_len <= 0) or (pb_len > len(iterator)):
            pb_len = len(iterator)
        
        self.pb_len = pb_len
        tqdm.__init__(self, iterable=iterator, desc=desc, total=self.pb_len)
        # self.pbar = tqdm(iterator, self.pb_len, desc=desc)

    def update_message(self, base_msg="", msg_dict={}):

        msg = base_msg
        for key, value in msg_dict.items():
            if (type(value) is float) or (type(value) is np.float64):
                msg = msg + "[{0}:{1:.4}] ".format(key, value)
            else:
            # if (type(value) is int) or (type(value) is str) or (type(value) is tuple):
                msg = msg + "[{0}:{1}] ".format(key, value)
            # print(msg)

        self.set_postfix_str(msg)
        self.refresh()

# class BatchSampler_V1(Sampler):
#     r"""Wraps another sampler to yield a mini-batch of indices.

#     Args:
#         sampler (Sampler): Base sampler.
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If ``True``, the sampler will drop the last batch if
#             its size would be less than ``batch_size``

#     Example:
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#     """

#     def __init__(self, sampler, batch_size, drop_last):
#         if not isinstance(sampler, Sampler):
#             raise ValueError("sampler should be an instance of "
#                              "torch.utils.data.Sampler, but got sampler={}"
#                              .format(sampler))
#         if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
#                 batch_size <= 0:
#             raise ValueError("batch_size should be a positive integeral value, "
#                              "but got batch_size={}".format(batch_size))
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#     def __iter__(self):
#         batch = []
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size