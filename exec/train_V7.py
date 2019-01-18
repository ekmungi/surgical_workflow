import sys

sys.path.append('/home/avemuri/DEV/src/surgical_workflow/')
# sys.path.append('/media/anant/dev/src/surgical_workflow/')

from dataloader.workflow_dataset import kFoldWorkflowSplit, WorkflowDataset
from model.workflow_resnet_model import ResFeatureExtractor
from utils.early_stopping import EarlyStopping
from utils.helpers import ModelCheckpoint, CumulativeMovingAvgStd, ProgressBar, Engine, create_plot_window, BestScore
from utils.optimizer_utils import get_optimizer

from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch

from batchgenerators.dataloading import SlimDataLoaderBase
import numpy as np
import time

from batchgenerators.transforms import MirrorTransform, GaussianNoiseTransform, NumpyToTensor, Compose
from batchgenerators.dataloading import MultiThreadedAugmenter

class WorkflowDataloader(SlimDataLoaderBase):
    def generate_train_batch(self):
        idxs = np.random.choice(len(self._data), self.batch_size, replace=True)
        images, labels = [], []
        for idx in idxs:
            i, l = self._data[idx]
            images.append(i)
            labels.append(l)
        images = np.vstack(images)
        labels = np.vstack(labels)
        return {'data': images, 'target': labels}



if __name__ == '__main__':

    dataset = WorkflowDataset(video_path='/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/COMPRESSED_0_05/TrainingSet/videos/Prokto1_cleaned_compressed_0_05.avi',
                                phase_path='file:///home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/COMPRESSED_0_05/TrainingSet/phase_annotations/Prokto1_cleaned_compressed_0_05.csv',
                                num_phases=14)




    tr_transforms = []
    # tr_transforms.append(MirrorTransform((0, 1)))
    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.3))
    tr_transforms.append(NumpyToTensor(keys=['data', 'target']))
    tr_transforms = Compose(tr_transforms)

    dataloader = WorkflowDataloader(data=dataset, batch_size=16, number_of_threads_in_multithreaded=None)

    mt_gen = MultiThreadedAugmenter(dataloader, tr_transforms, 8)

    for epoch in range(5):
        t0 = time.time()
        for b in range(100):
            data_dict = next(mt_gen)
            data = data_dict['data']
            labels = data_dict['target']
        t1 = time.time()
        print('duration: {}'.format(t1-t0))