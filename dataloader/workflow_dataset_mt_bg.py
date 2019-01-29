import sys, os

# sys.path.append('/home/avemuri/DEV/src/surgical_workflow/')
sys.path.append('/media/anant/dev/src/surgical_workflow/')


from glob import glob
import numpy as np
import pandas as pd
import imageio
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets import DatasetFolder

from batchgenerators.dataloading import SlimDataLoaderBase
from batchgenerators.dataloading import MultiThreadedAugmenter

from sklearn.model_selection import KFold

from utils.helpers import kfold_split, fast_collate_nvidia_apex

from tqdm import tqdm

import imgaug as ia

import time

from tqdm import trange



class WorkflowDataset(DatasetFolder):

    def __init__(self,
                 video_path,
                 phase_path,
                 num_phases,
                 instrument_path=None,
                 input_transform=None):
        """
        Specifically desiged for loading data for worklow analysis
        in surgical videos.

        Arguments
        ---------
        video_path : path to the location of the video
        phase_path : path to the location of the phase annotations
        instrument_path : path to the location of the instrument annotations
        
        input_transform : class which implements a __call__ method
            tranform(s) to apply to inputs during runtime loading

        """
        self.video_path = video_path
        self.phase_path = phase_path
        self.instrument_path = instrument_path
        self.num_inputs = 1
        self.num_phases = num_phases
        self.transform = input_transform

        # Run this once to ensure the one_hot_encoder model is created
        # self.one_hot_encoder = _fit_one_hot_encoder_(np.arange(self.num_phases))
        self._process_input_()

        # self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
        # self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else _pass_through


    def _process_input_(self):

        self.video = imageio.get_reader(self.video_path)
        # self.phase_data = self.one_hot_encoder.transform(pd.read_csv(self.phase_path[self.current_video]).values[:,1].reshape(-1, 1)).toarray().view(np.float32)
        self.phase_data = pd.read_csv(self.phase_path, header=None, index_col=False).values[:,1].reshape(-1, 1)

        if self.instrument_path is not None:
            self.instrument_annotation_data = pd.read_csv(self.instrument_path, index_col=0)
        else:
            self.instrument_annotation_data = None

        
    def __getitem__(self, index):
        """
        Index the dataset and return the input + target
        """

        if index < len(self.video):
            try:
                current_frame = self.video.get_data(index)
            except imageio.core.format.CannotReadFrameError:
                print('CannotReadFrameError: Trying to access [{0}]'.format(index))
                print('Trying to access the following video: {0}'.format(self.video_path))
                exit
        else:
            print('ERROR: Index out of range.')
            exit
            
        # print(current_frame.shape)
        current_frame = Image.fromarray(current_frame)

        if self.transform is not None:
            current_frame = self.transform(current_frame)


        # transformed_image = [self.input_transform[0](current_frame)]

        # transformed_image = Transformations.perform_image_transformations(current_frame, 
        #                                                 transformations=self.transformations)
        
        if self.instrument_annotation_data is not None:
            try:
                instrument_annotation = self.instrument_annotation_data.loc[index].values
            except:
                nearest_index = _find_nearest_(self.instrument_annotation_data.index.values, index)
                instrument_annotation = self.instrument_annotation_data.loc[nearest_index].values
            # return (transformed_image[0], self.phase_data[index, :], instrument_annotation)
            return (current_frame, self.phase_data[index].squeeze(), instrument_annotation)
        else:
            # return (transformed_image[0], self.phase_data[index, :])
            instrument_annotation = None
            return (current_frame, self.phase_data[index].squeeze())

        

    def __len__(self):
        # return self.video._get_meta_data(0)['nframes'] # SAME THING
        return len(self.video)



def load_batches(data, batch_size, n_batches):
    for iBatch in range(n_batches):
        dataset_choice = np.random.choice(len(data), batch_size, replace=True)
        images, phase_annotations = [], []
        for idx_dataset in dataset_choice:
            # print("\n\n\n{0}\n\n\n".format(len(self._data[idx_dataset])))
            frame_choice = np.random.choice(len(data[idx_dataset]), replace=True)
            # print("\n\n\n{0}\n\n\n".format(frame_choice))
            image, phase = data[idx_dataset][frame_choice]
            images.append(image)
            phase_annotations.append(phase)
        
        images = np.stack(images, axis=0)
        # np.array(images, dtype=np.uint8)
        batch = ia.Batch(images=images, data=np.array(phase_annotations))

        yield batch




class WorkflowBatchGenerator(SlimDataLoaderBase):
    def __init__(self, data, batch_size):

        super(WorkflowBatchGenerator, self).__init__(data, batch_size)
        
        self.total_len = 0
        self._data = data
        for data_item in self._data:
            self.total_len += len(data_item)
        # self.n_batches = n_batches


    def generate_single_batch(self):
        dataset_choice = np.random.choice(len(self._data), self.batch_size, replace=True)
        images, phase_annotations = [], []
        for idx_dataset in dataset_choice:
            # print("\n\n\n{0}\n\n\n".format(len(self._data[idx_dataset])))
            frame_choice = np.random.choice(len(self._data[idx_dataset]), replace=True)
            # print("\n\n\n{0}\n\n\n".format(frame_choice))
            image, phase = self._data[idx_dataset][frame_choice]
            images.append(image)
            phase_annotations.append(phase)
        
        images = np.stack(images, axis=0).astype(np.uint8)
        # images = np.moveaxis(np.stack(images, axis=0).astype(np.float32), [0,1,2,3], [0,2,3,1])
        return {'data': images, 'target': np.array(phase_annotations)}


    def load_batches(self):
        # for iBatch in range(self.n_batches):
        #     yield self.generate_single_batch()
        return


    def __iter__(self):
        return self 

    def __next__(self):
        return self.generate_single_batch()

    def __len__(self):
        return self.total_len


class kFoldWorkflowSplitMT(DatasetFolder):

    def __init__(self, base_path, image_transform=None, n_folds=4,
                    video_extn='.avi', shuffle=False, num_phases=14, 
                    batch_size=4, num_workers=1, 
                    use_custom_collate=False, video_folder='videos', phase_folder='phase_annotations'):
        self.video_base_path = os.path.join(base_path, video_folder)
        self.phase_base_path = os.path.join(base_path, phase_folder)

        # print("\n\n{}\n\n".format(self.video_base_path))
        # print("\n\n{}\n\n".format(self.phase_base_path))

        self.instrument_base_path = None#os.path.join(base_path, 'instrument_annotations')
        self.image_transform = image_transform
        self.video_extn = video_extn
        self.num_phases = num_phases
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.n_folds = n_folds
        self.n = 0
        self._process_input_()
        self.coutom_collate_fn = default_collate
        # self.n_batches = n_batches
        if use_custom_collate:
            self.coutom_collate_fn = fast_collate_nvidia_apex

        self.train_bg_augmenter = None
        self.valid_bg_augmenter = None

        self.n_max_generators = 3

        # print("\n\nBuilding generator list...")
        # t0 = time.time()
        # self._build_generator()
        # t1 = time.time()
        # print("\n\n\nDuration: {0}\n\n\n".format(np.round(t1-t0,2)))
        

    def _process_input_(self):
        phase_list_paths = glob(os.path.join(self.phase_base_path, '*.csv'))
        self.phase_list = []
        self.video_list = []
        self.instrument_list = []
        for phase_path in phase_list_paths:
            _, file_name = os.path.split(phase_path)
            file_name, _ = os.path.splitext(file_name)
            video_path = os.path.join(self.video_base_path, file_name+self.video_extn)
            if self.instrument_base_path is not None:
                instrument_path = os.path.join(self.instrument_base_path, file_name+".csv")
            else:
                instrument_path = None
            if os.path.isfile(video_path) & os.path.isfile(phase_path):
                self.phase_list.append(phase_path)
                self.video_list.append(video_path)
                self.instrument_list.append(instrument_path)


        self.kf_split = KFold(n_splits=self.n_folds, shuffle=self.shuffle)
        self.kf_split.get_n_splits(self.video_list)

        self.video_list = np.array(self.video_list)
        self.phase_list = np.array(self.phase_list)
        self.instrument_list = np.array(self.instrument_list)

        self.kf_split_iter = iter(self.kf_split.split(self.video_list))


    def _build_generator(self):
        self.train_batchgenerator = None
        self.valid_batchgenerator = None
        # for iFold in trange(self.n_folds):
        train_index, valid_index = next(self.kf_split_iter)
        # print(train_index, valid_index)

        train_datasets = self._accumulate_datasets_(train_index)
        valid_datasets = self._accumulate_datasets_(valid_index)

        self.train_batchgenerator = WorkflowBatchGenerator(data=train_datasets, batch_size=self.batch_size)
        self.valid_batchgenerator = WorkflowBatchGenerator(data=valid_datasets, batch_size=self.batch_size)

        # train_bg_augmenter = MultiThreadedAugmenter(train_batchgenerator, self.image_transform, 
        #                                             self.num_workers, pin_memory=True)
        # valid_bg_augmenter = MultiThreadedAugmenter(valid_batchgenerator, self.image_transform, 
        #                                             self.num_workers, pin_memory=True)

            

    def __iter__(self):
        return self

    def __next__(self):

        if self.n <= self.n_folds:
            # t0 = time.time()
            self._build_generator()
            # t1 = time.time()
            # print("\n\n\nDuration: {0} {1}\n\n\n".format(np.round(t1-t0,2), self.n))

            if self.train_bg_augmenter is None:
                self.train_bg_augmenter = MultiThreadedAugmenter(self.train_batchgenerator, self.image_transform, 
                                                                    self.num_workers, pin_memory=True)
            else:
                self.train_bg_augmenter.set_generator(self.train_batchgenerator)

            if self.valid_bg_augmenter is None:
                self.valid_bg_augmenter = MultiThreadedAugmenter(self.valid_batchgenerator, self.image_transform, 
                                                                    self.num_workers, pin_memory=True)
            else:
                self.valid_bg_augmenter.set_generator(self.valid_batchgenerator)

            self.n += 1            
        
            return self.train_bg_augmenter, self.valid_bg_augmenter
        else:
            raise StopIteration

    def __len__(self):
        return self.n_folds


    def _accumulate_datasets_(self, select_index):
        
        
        datasets = []
        for video_path, phase_path, instrument_path in zip(self.video_list[select_index], 
                                                            self.phase_list[select_index],
                                                            self.instrument_list[select_index]):
            datasets.append(WorkflowDataset(video_path, phase_path, self.num_phases, 
                                                    instrument_path, input_transform=None))

        return datasets

    
    # def get_length(self,):
        


        

def _fit_one_hot_encoder_(label):
    encoder = OneHotEncoder(dtype=np.float32)
    _ = encoder.fit(label.reshape(-1,1))
    # print('The labels are: {}'.format(np.unique(label)))
    return encoder

def _one_hot_encoder_(label):
    encoder = OneHotEncoder(dtype=np.float32)
    label_1hot = encoder.fit_transform(label.reshape(-1,1))
    print('The labels are: {}'.format(np.unique(label)))
    return label_1hot

def _find_nearest_(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



## ==================================================================== ##

def test_kfold_split():
    torch.manual_seed(42)
    np.random.seed(42)
    
    base_path = '/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/'
    image_transform = transforms.Compose([transforms.Resize((320,180)),
                                            transforms.ToTensor()])
    kfoldWorkflowSet = kFoldWorkflowSplit(base_path, 
                                            image_transform=image_transform,
                                            video_extn='.avi', shuffle=True,
                                            n_folds=18, num_phases=14,
                                            batch_size=16, num_workers=1)

    train_loader, valid_loader = next(kfoldWorkflowSet)
    
    for train_loader, valid_loader in tqdm(kfoldWorkflowSet):
        for img, phase in tqdm(valid_loader):
            if img.shape[1:] == (3,320,180):
                pass
            else:
                print("TROUBLE!!")
                

if __name__ == '__main__':
    test_kfold_split()