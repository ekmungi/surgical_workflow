import sys

# sys.path.append('/home/avemuri/DEV/src/surgical_workflow/')
sys.path.append('/media/anant/dev/src/surgical_workflow/')

from dataloader.workflow_dataset_mt import kFoldWorkflowSplitMT
from model.workflow_resnet_model import ResFeatureExtractor
from utils.early_stopping import EarlyStopping
from utils.helpers import ModelCheckpoint, CumulativeMovingAvgStd, ProgressBar, Engine, create_plot_window, BestScore
from utils.optimizer_utils import get_optimizer

from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
import time

# from batchgenerators.dataloading import SlimDataLoaderBase
# from batchgenerators.transforms import RangeTransform, NumpyToTensor, SimulateLowResolutionTransform, Compose
# from batchgenerators.dataloading import MultiThreadedAugmenter
# from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform

from tqdm import tqdm, trange

from imgaug import augmenters as iaa

# class WorkflowDataloader(SlimDataLoaderBase):
#     def generate_train_batch(self):
#         data = np.random.choice(len(self._data), self.batch_size, replace=True)
#         images, labels = [], []
#         for idx in idxs:
#             i, l = self._data[idx]
#             images.append(i)
#             labels.append(l)
#         images = np.vstack(images)
#         labels = np.vstack(labels)
#         return {'data': images, 'target': labels}



if __name__ == '__main__':

    # dataset = WorkflowDataset(video_path='/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/COMPRESSED_0_05/TrainingSet/videos/Prokto1_cleaned_compressed_0_05.avi',
    #                             phase_path='/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/COMPRESSED_0_05/TrainingSet/phase_annotations/Prokto1_cleaned_compressed_0_05.csv',
    #                             num_phases=14)

    EPOCHS = 1
    ITERATIONS = 8
    FOLDS = 1
    # image_transform = []
    # tr_transforms.append(MirrorTransform((0, 1)))
    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.3))
    # image_transform.append()

    image_transform = iaa.Sequential([iaa.Scale(0.5)])
    
    # Compose([SimulateLowResolutionTransform(zoom_range=[0.25, 0.25]),
    #                             #RangeTransform(rnge=(0,1), data_key='data'),
    #                             NumpyToTensor(keys=['data', 'target'])])
    
    
    kfoldWorkflowSet = kFoldWorkflowSplitMT('/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/', 
                                            image_transform=image_transform,
                                            video_extn='.avi', shuffle=True,
                                            n_folds=21, num_phases=14,
                                            batch_size=32, 
                                            num_workers=4)

    eta = CumulativeMovingAvgStd()

    for iFold in range(FOLDS):
        train_gen, valid_gen = next(kfoldWorkflowSet)
        for epoch in range(EPOCHS):
            t0 = time.time()
            for iteration in trange(ITERATIONS):
                batch = train_gen.get_batch()
                images = batch.images_aug
                phase_annotations = batch.data
                print (images.shape, phase_annotations.shape)
            t1 = time.time()
            print('Fold {0}, Epoch {1} : {2}'.format(iFold, epoch, np.round(t1-t0,2)))
        time.sleep(0.01)
        if iFold+1 == FOLDS:
            break
        del(train_gen)
        del(valid_gen)

    # with ProgressBar(kfoldWorkflowSet, desc="FOLDS", pb_len=FOLDS) as folds_pbar:
    #     epoch_msg_dict = {}

    #     for iFold, (train_gen, valid_gen) in enumerate(folds_pbar):

    #         with ProgressBar(range(EPOCHS), desc="EPOCHS", pb_len=EPOCHS) as epoch_pbar:

    #             for epoch in epoch_pbar:

    #                 with ProgressBar(train_gen, desc="TRAIN", pb_len=ITERATIONS) as iteration_pbar:

    #                     t0 = time.time()
    #                     for iteration, data_dict in enumerate(iteration_pbar):
            
    #                         data = data_dict['data']
    #                         labels = data_dict['target']

    #                         if iteration_pbar.step(iteration+1):
    #                             break
            
    #                     t1 = time.time()
    #                     eta.update(np.round(t1-t0,2))
    #                     epoch_msg_dict['ETA'] = eta.get_value()[0]
    #                     folds_pbar.update_message(msg_dict=epoch_msg_dict)

    #                 time.sleep(0.01)
    #                 if epoch_pbar.step(epoch+1):
    #                     break

    #         del(train_gen)
    #         del(valid_gen)
    #         time.sleep(0.01)

    #         if folds_pbar.step(iFold+1):
    #             break

            
            
            
                
                #data_dict = next(train_gen)
                

                # if (iteration+1) == len(iteration_pbar):
                #     iteration_pbar.refresh()
                #     iteration_pbar.close()
                #     break

            
            
            #print('Epoch {1}: {2}'.format(iFold, epoch, np.round(t1-t0,2)))

            # if (epoch+1) == len(epoch_pbar):
            #     epoch_pbar.refresh()
            #     epoch_pbar.close()
            #     break

        # del(train_gen)
        # del(valid_gen)
        
        # if (iFold+1) == len(folds_pbar):
        #     folds_pbar.refresh()
        #     folds_pbar.close()
        #     break

    
    # dataset = WorkflowDataset(video_path='/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/videos/Prokto1_cleaned_compressed_0_05.avi',
    #                             phase_path='/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/phase_annotations/Prokto1_cleaned_compressed_0_05.csv',
    #                             num_phases=14)


    

    # dataloader = WorkflowDataloader(data=dataset, batch_size=32, number_of_threads_in_multithreaded=None)

    # mt_gen = MultiThreadedAugmenter(dataloader, tr_transforms, 12)

    # N = 300
    # for epoch in range(5):
    #     t0 = time.time()
    #     for b in range(N):
    #         data_dict = next(mt_gen)
    #         data = data_dict['data']
    #         labels = data_dict['target']
    #     t1 = time.time()
    #     print('Epoch {0}: {1}'.format(epoch, np.round(t1-t0,2)))