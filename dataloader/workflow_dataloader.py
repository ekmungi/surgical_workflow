# import sys
# import os

# import numpy as np
# # import matplotlib.pyplot as plt
# from itertools import cycle
# import pandas as pd
# import timeit
# import math
# import time

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from imgaug import augmenters as iaa


# # from trixi.logger import PytorchVisdomLogger as pvl

# # import torchsample
# # from torchsample import WorkflowDataset, WorkflowListDataset
# # from torchsample.transforms import Compose, ChannelsFirst, TypeCast, ResizePadArray, AddChannel, RangeNormalize
# # from torchsample.transforms import ImgAugTranform, ImgAugCoTranform

# from utils.helpers import kfold_split
# from utils.tranformations import ImgAugTransform
# from dataloader.workflow_dataset import WorkflowListDataset



# import imageio
# from PIL import Image
# from tqdm import tqdm

# # from giana.utils import dice_loss
# # from concurrent.futures import ThreadPoolExecutor
# # from imgaug import BackgroundAugmenter, BatchLoader, Batch

# # from multiprocessing import Pool



# # Imgaug tranforms to get the proper data augmentation
# # iaa.Sequential([
# #     iaa.Scale((224, 224)),
# #     iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
# #     iaa.Fliplr(0.5),
# #     iaa.Affine(rotate=(-20, 20), mode='symmetric'),
# #     iaa.Sometimes(0.25,
# #                   iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
# #                              iaa.CoarseDropout(0.1, size_percent=0.5)])),
# #     iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
# # ])


# # add_image_transform_list = [
# #     iaa.Sometimes(0.5, [iaa.GaussianBlur(sigma=(0.5, 2.0)),
# #                         iaa.Multiply((0.8, 1.2)),
# #                         iaa.ContrastNormalization(
# #         (0.75, 1.5)),
# #         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),
# #                                   per_channel=True),
# #         # iaa.MotionBlur(size=(5, 10))]),
# #         iaa.MotionBlur(k=15, angle=[-45, 45])]),
# #     iaa.Fliplr(0.5)
# # ]

# # add_joint_transform_list = [

# #                             ]


# BATCH_SIZE = 24
# IMAGE_SIZE = 320
# NUM_PHASES = 14
# NUM_WORKERS = 12

# def plot_image_label(images, labels):

#     # fig = plt.figure(figsize=(15, 7))

#     count = 0
#     for image, label in zip(images, labels):

#         fig = plt.figure(figsize=(5, 5))

#         timer = fig.canvas.new_timer(interval=1000)
#         timer.add_callback(close_event)

#         label = label[:, :, 0]
#         max_val = np.max([np.max(np.max(label)), 1.0])
#         color_delta = [100, -80, -80]
#         image[:, :, 0] = np.clip(
#             image[:, :, 0] + color_delta[0] * (label/max_val), 0, 255)
#         image[:, :, 1] = np.clip(
#             image[:, :, 1] + color_delta[1] * (label/max_val), 0, 255)
#         image[:, :, 2] = np.clip(
#             image[:, :, 2] + color_delta[2] * (label/max_val), 0, 255)

#         #print("\nImage ==> Max:{0}, Min:{1}".format(np.max(image), np.min(image)))
#         #print("Label ==> Max:{0}, Min:{1}\n".format(np.max(label), np.min(label)))
#         ax = plt.subplot(2, 1, 1)
#         max_val = np.max(image)
#         plt.imshow((image*255.0/max_val).astype(np.uint8))

#         ax = plt.subplot(2, 1, 2)
#         plt.imshow(label, cmap='gray')

#         plt.tight_layout()
#         ax.set_title('Sample #{}'.format(count+1))
#         ax.axis('off')

#         count += 1
#         timer.start()
#         plt.show()
#     return None


# def close_event():
#     plt.close() 



# def get_dataloaders(base_path, shuffle=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
#                     n_threads=4, n_folds=4):

#     torch.manual_seed(42)
#     ## Transformations ############################################################
#     # TorchVision for getting the right format
#     # These transforms are applied by pytorch dataloader directly
    
#     # aug_transform = iaa.Sequential([iaa.Scale(1/6)])
#     # image_transform = transforms.Compose([ImgAugTransform(aug_transform),
#     #                                         transforms.ToTensor()])
#     image_transform = transforms.Compose([transforms.Resize((320,180)),
#                                             transforms.ToTensor()])

#     video_base_path = os.path.join(base_path, 'videos')
#     phase_base_path = os.path.join(base_path, 'phase_annotations')
#     instrument_base_path = None

#     kf, video_list, phase_list, instrument_list = kfold_split(video_base_path, 
#                                                                 phase_base_path, 
#                                                                 instrument_base_path, 
#                                                                 video_extn='.avi',
#                                                                 n_splits=4)

#     for train_index, test_index in kf_split.split(video_list):

#     # if n_folds > 0:    

#     instrument_path = None

#     train_workflowset = WorkflowListDataset(video_base_path=video_path,
#                                           phase_base_path=phase_path,
#                                           instrument_base_path=instrument_path,
#                                           num_phases=NUM_PHASES,
#                                           video_extn=".avi",
#                                           image_transform=image_transform)

#     valid_workflowset = WorkflowListDataset(video_base_path=video_path,
#                                           phase_base_path=phase_path,
#                                           instrument_base_path=instrument_path,
#                                           num_phases=NUM_PHASES,
#                                           video_extn=".avi",
#                                           image_transform=image_transform)

#     train_loader = DataLoader(train_workflowset, batch_size, shuffle,  num_workers=n_threads)
#     valid_loader = DataLoader(valid_workflowset, batch_size, shuffle,  num_workers=n_threads)

#     return train_loader, valid_loader


# def train():

#     transformations, train_loader = get_dataloaders()
#     thread_pool = Pool()
#     iterator = iter(train_loader)

#     n_iter = 2
#     idx = 0

#     for images, phase_annotations in train_loader:

#         images, phase_annotations = transformations.apply_transform(
#             (images, phase_annotations))
#         print("{0}: {1}   {2}".format(
#             idx+1, images.size(), phase_annotations.size()))

#         idx += 1

#         # ****** Perform training here ******

#         if idx == n_iter:
#             break


# def test_data_loader():

#     base_path = "/home/anant/Dropbox/Temp/endovis_sample/"

#     # imgaug_transform, workflow_loader = get_dataloaders(base_path, shuffle=True, loop=True)
#     video_base_path = os.path.join(base_path, 'videos')
#     phase_base_path = os.path.join(base_path, 'phase_annotations')
#     instrument_base_path = None

#     NUM_PHASES = 14
#     IMAGE_SIZE = 320

#     image_transform = transforms.Compose([
#         ResizePadArray(IMAGE_SIZE),
#         #ChannelsFirst(),
#         #TypeCast('float')
#     ])

#     # image_transform = None

#     workflowset = WorkflowListDataset(video_base_path=video_base_path,
#                                       phase_base_path=phase_base_path,
#                                       instrument_base_path=instrument_base_path,
#                                       num_phases=NUM_PHASES,
#                                       video_extn='.avi',
#                                       image_transform=image_transform)

#     workflow_loader = DataLoader(workflowset, BATCH_SIZE, False)

#     iterations = 0
#     for images, phase_annotations in workflow_loader:
#         print("Iteration: {0}, Image shape: {1}, Phase: {2}".format(iterations, images.shape, phase_annotations))
#         iterations += 1

#     # for idx in [3855, 5420, 2540]:
#     #     images, phase_annotations = workflowset[idx]
#     #     print("Image shape: {0}, Phase: {1}".format(images.shape, phase_annotations))

#     print("Total iterations: {0}".format(iterations))
#     print('Done')


#     # plt.show()
# if __name__ == "__main__":
#     # image_gt_file_list_all = '/home/avemuri/Dropbox/Temp/image_gt_data_file_list_all_640x640_linux.csv'
#     # train()
#     test_data_loader()
