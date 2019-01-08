import sys, os

# sys.path.append('/home/avemuri/DEV/projects/endovis2018-challenge/')
sys.path.append('/media/anant/dev/src/endovis/')

import numpy as np
import imageio
from glob import glob
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from workflow.dataloader.workflow_dataset import WorkflowDataset
from workflow.utils.helpers import ProgressBar

import matplotlib.pyplot as plt

from tqdm import tqdm

FONT = cv2.FONT_HERSHEY_SIMPLEX

def play_video(video_file, phase_file):

    image_transform = ToTensor()
    dataset = WorkflowDataset(video_file, phase_file, 14, None, image_transform)
    dataloader = DataLoader(dataset, batch_size=18, shuffle=True, num_workers=4)
    pbar = ProgressBar(dataloader, "Frame")
    msg_dict={}
    for i, (image, phase) in enumerate(pbar):
        image.to(torch.device('cuda:1'))
        phase.to(torch.device('cuda:1'))
        # image = np.moveaxis(image.numpy(), [0,1,2,3], [0,3,1,2])
        # ax.imshow(image)
        # ax.set_title(str(phase.numpy()), fontsize=14)
        # plt.pause(.01)
        # plt.draw()
        msg_dict['ID'] = i
        msg_dict['shape'] = image.shape
        pbar.update_message(msg_dict=msg_dict)

    # plt.show()





if __name__ == "__main__":
    
    phase_base_path = '/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/phase_annotations/'
    video_base_path = '/home/anant/data/endovis/COMPRESSED_0_05/TrainingSet/videos/'


    phase_list_paths = glob(os.path.join(phase_base_path, '*.csv'))
    phase_list = []
    video_list = []
    instrument_list = []
    for phase_path in phase_list_paths:
        _, file_name = os.path.split(phase_path)
        file_name, _ = os.path.splitext(file_name)
        video_path = os.path.join(video_base_path, file_name+'.avi')
        if os.path.isfile(video_path) & os.path.isfile(phase_path):
            phase_list.append(phase_path)
            video_list.append(video_path)

    

    for video_path, phase_path in zip(video_list, phase_list):
        # fig = plt.figure(tight_layout=True)
        # ax = fig.gca()
        print(video_path)
        play_video(video_path, phase_path)
        # plt.close()


