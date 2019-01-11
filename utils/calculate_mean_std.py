import sys, os

sys.path.append('/home/avemuri/DEV/src/surgical_workflow/')
# sys.path.append('/media/anant/dev/src/surgical_workflow/')

import numpy as np
import imageio
from glob import glob
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from dataloader.workflow_dataset import WorkflowDataset
from utils.helpers import ProgressBar, CumulativeMovingAvgStd

import matplotlib.pyplot as plt

from tqdm import tqdm

from argparse import ArgumentParser

FONT = cv2.FONT_HERSHEY_SIMPLEX

def play_video(video_file, phase_file, avg=None):

    image_transform = ToTensor()
    dataset = WorkflowDataset(video_file, phase_file, 14, None, image_transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=12)
    pbar = ProgressBar(dataloader, "Frame")
    msg_dict={}
    if avg is None:
        avg = [CumulativeMovingAvgStd(), CumulativeMovingAvgStd(), CumulativeMovingAvgStd()]
    for i, (image, phase) in enumerate(pbar):

        avg[0].update(np.mean(image[:,0,:,:].numpy().flatten()))
        avg[1].update(np.mean(image[:,1,:,:].numpy().flatten()))
        avg[2].update(np.mean(image[:,2,:,:].numpy().flatten()))
        # image.to(torch.device('cuda:0'))
        # phase.to(torch.device('cuda:0'))
        # image = np.moveaxis(image.numpy(), [0,1,2,3], [0,3,1,2])
        # ax.imshow(image)
        # ax.set_title(str(phase.numpy()), fontsize=14)
        # plt.pause(.01)
        # plt.draw()
        msg_dict['ID'] = i
        msg_dict['shape'] = image.shape
        pbar.update_message(msg_dict=msg_dict)

    return avg

    # plt.show()





if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--all", action="store_true", dest="all",
                        help="For complete dataset", default=False)
    options = parser.parse_args()
    
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

    if options.all:
        print("Computing for all the data...")
        avg = [CumulativeMovingAvgStd(), CumulativeMovingAvgStd(), CumulativeMovingAvgStd()]
    else:
        avg = None

    for video_path, phase_path in zip(video_list, phase_list):
        # fig = plt.figure(tight_layout=True)
        # ax = fig.gca()
        print(video_path)
        if options.all:
            avg = play_video(video_path, phase_path, avg)
        else:
            avg = play_video(video_path, phase_path, None)

        print("[{0},{1},{2}]".format(avg[0].get_value()[0], avg[1].get_value()[0], avg[2].get_value()[0]))
        print("[{0},{1},{2}]".format(avg[0].get_value()[1], avg[1].get_value()[1], avg[2].get_value()[1]))
        # plt.close()



def all_data():
    avg_val = np.array([[0.2979509885481896,0.17308739014483032,0.19925256966313049],
                        [0.349034960767266,0.19069608288269418,0.18698055832597352],
                        [0.28900872095268054,0.1530121886024062,0.16504192058986583],
                        [0.3345794373578986,0.19162272370646516,0.21928634399494523],
                        [0.35560642601105197,0.21297440507955714,0.21632554087983943],
                        [0.33341929652061053,0.18866799917098456,0.2044329839289613],
                        [0.4441005945883869,0.26292820996339916,0.2580759356695542],
                        [0.3086047727515092,0.20322882774929107,0.25467233850231635],
                        [0.4425088331688362,0.2819976564023565,0.25845792659022776],
                        [0.381047637926208,0.21716850861109824,0.19789546409261566],
                        [0.33915850630772293,0.19967223427965672,0.22548753787812437],
                        [0.30329327101499326,0.19234474828958056,0.19571267851615995],
                        [0.3404103780434991,0.1953692977075224,0.18479481582790117],
                        [0.2881239824849713,0.1648862101067744,0.1532279201145468],
                        [0.4343195655783595,0.22631614055610516,0.2643403977875551],
                        [0.4473215542354825,0.267907258539887,0.3563405303126674],
                        [0.48992207407237587,0.29514879214906425,0.2837699294447183],
                        [0.3565762483500711,0.20660845547110407,0.21459466392334553],
                        [0.3150979598917936,0.20106548040651426,0.21216227431890317],
                        [0.34448529881078765,0.19686808998800553,0.18726468442437322],
                        [0.35732362770178966,0.2129724355712422,0.21593554763426864],
                        [0.2967713228222723,0.17808816416309628,0.18639637782609803],
                        [0.4043220769352662,0.2455857328000994,0.2893021505031287],
                        [0.3841529554491056,0.27394649517489883,0.39419076829801686]])


    std_val = np.array([[0.01731367416913888,0.025922903954991406,0.033806948458727844],
                        [0.018096556296566355,0.023062247131786475,0.03246522760917757],
                        [0.014440707870165954,0.02783438452231958,0.031775475796148374],
                        [0.016320273316265486,0.028143717621738992,0.032289194490953216],
                        [0.019038916878446038,0.02358272768750487,0.025966877636426484],
                        [0.01508455312259961,0.02231802026903126,0.02808538047266429],
                        [0.027122769122272364,0.03759375124646213,0.048922720326381085],
                        [0.01808807215461924,0.020009470208492062,0.029108898733848403],
                        [0.02538816994837612,0.031710953578103704,0.03800484949604527],
                        [0.027335481770621547,0.02347536463414633,0.025619624745173132],
                        [0.01893181055277738,0.026117232256263872,0.03717054798643847],
                        [0.015642630070037872,0.018041648005744402,0.02095576670405289],
                        [0.01638016459732749,0.01991192357908367,0.027558356265522953],
                        [0.013782966604773652,0.020431811598087907,0.03012192453522083],
                        [0.033036907060187276,0.026691884007316154,0.04096263011033144],
                        [0.03848236002990661,0.03571994364932396,0.05259974881041798],
                        [0.029499435216157797,0.029245393349093884,0.0455725323142051],
                        [0.018094551409480793,0.021627395219183856,0.0275028584428187],
                        [0.010989827655998919,0.014259976733252033,0.021727514663267535],
                        [0.029731465385714913,0.026628681517818525,0.029940105356119563],
                        [0.015981713506240922,0.026877499228409145,0.03820217649743541],
                        [0.013996671130173126,0.017596296079664918,0.024072409310358655],
                        [0.03721219921671407,0.031251607166788764,0.035747902909406874],
                        [0.036196666109185746,0.02467018596063735,0.04415712123777249]])

    tmp_avg = np.mean(avg_val,axis=0)
    tmp_avg = np.repeat(np.expand_dims(tmp_avg,axis=1), len(avg_val), axis=1).transpose()



    # plt.plot(tmp[:,0])
    # plt.plot(tmp[:,1])
    # plt.plot(tmp[:,2])
    # plt.plot(tmp_avg[:,0])
    # plt.plot(tmp_avg[:,1])
    # plt.plot(tmp_avg[:,2])
    # plt.show()