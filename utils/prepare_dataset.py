import numpy as np
import imageio
import pandas as pd
from tqdm import tqdm
import cv2

import os, sys
from glob import glob

from moviepy.editor import VideoFileClip, concatenate_videoclips



def create_folders_if_necessary(folder_path):
    if not os.path.exists(folder_path):
        print("\nCREATING FOLDER...", folder_path, "\n")
        os.makedirs(folder_path)

def sample_with_replacement(array, n_samples):
    return np.random.choice(array, n_samples)

def sample_without_replacement(array, n_samples):
    return np.random.choice(array, n_samples, replace=False)

def remove_blue_frames(video_path, phase_path, out_video, out_phase, n_samples=None):

    '''
    in_path: is the input folder path. Expected higher level folder containing two folders 'video' and 'phase'
    out_path: is the output folder path which will have 'video' and 'phase' folders with the cleaned data
    '''

    # video_local_dir = "videos"
    # phase_local_dir = "phase_annotations"

    # video_path = os.path.join(in_path, video_local_dir)
    # phase_path = os.path.join(in_path, phase_local_dir)


    
    video = imageio.get_reader(video_path)
    phase_data = pd.read_csv(phase_path, header=None).values[:,1]

    N = len(video)
    n_bins = 3
    if n_samples is None:
        step_size = int(N/N)
    else:
        step_size = int(N/n_samples)
    # fig, ax = plt.subplots(3,1)
    x_val = np.array(range(0,N,step_size))
    y_val = np.zeros(shape=(x_val.shape[0]))
    count = 0

    writer = imageio.get_writer(out_video, fps=30, codec='mjpeg', quality=10, pixelformat='yuvj444p')
    index_all = []

    for frame_id in tqdm(range(0,N,step_size)):
        frame = video.get_data(frame_id)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ax[0].imshow(frame)
        hist = np.bincount(gray.ravel(), minlength=256)
        hist_count = max(hist) - np.sum(hist[hist<max(hist)])
        y_val[count] = hist_count
        if hist_count<=0:
            writer.append_data(frame)
            index_all.append(frame_id)
        count += 1


    writer.close()


    # blue_frames_loc = y_val<=0
    # # _, out_file_name = os.path.split(video_path)
    # # out_file_name, out_file_extension = os.path.splitext(out_file_name)
    # # out_video = os.path.join(out_path, out_file_name+"_cleaned"+out_file_extension)
    # writer = imageio.get_writer(out_video, fps=30, codec='mjpeg', quality=10, pixelformat='yuvj444p')

    # count = 0
    # index_all = []
    # for frame_id in tqdm(range(0,N,step_size)):
    #     if blue_frames_loc[count]:
    #         img = video.get_data(frame_id)
    #         writer.append_data(img)
    #         index_all.append(frame_id)
    #     count += 1

    # writer.close()

    # return index_all
    frame_ids = np.array(list(range(len(index_all))), dtype=np.int)
    phase_data_cleaned = np.vstack((frame_ids, np.array(phase_data[index_all], dtype=np.int))).transpose()
    np.savetxt(out_phase, phase_data_cleaned, delimiter=',', fmt='%d')


def compress_video_for_pretraining(video_path, phase_path, out_video, out_phase, fraction=0.1):

    video = imageio.get_reader(video_path)
    phase_data = pd.read_csv(phase_path, header=None).values[:,1]

    # Get the count of No. of frames in each phase

    first_element = phase_data[0]
    phase_seq_counts = []
    count_idx = 0
    for num in phase_data:
        if num==first_element:
            count_idx+=1
        else:
            phase_seq_counts.append(count_idx)
            count_idx = 1
            first_element = num

    phase_seq_counts.append(count_idx)
    normalized_phase_seq_counts = phase_seq_counts/np.sum(phase_seq_counts)
    
    max_phase_seq_count = np.max(phase_seq_counts)
    n_selected = int(fraction*max_phase_seq_count)

 
    index_all = []
    increment = 0
    total_selected_frames = 0
    for i, count_val in enumerate(phase_seq_counts):
        index = list(range(count_val))
        
        if count_val>=n_selected:
            selected_index = sample_without_replacement(index, n_selected)
            total_selected_frames += n_selected
            # print("Sampled WITHOUT replacement {0} frames".format(selected_index.shape[0]))
        elif count_val<0.5*n_selected:
            selected_index = sample_without_replacement(index, count_val)
            total_selected_frames += count_val
            # print("Sampled WITHOUT replacement {0} frames".format(selected_index.shape[0]))
        else:
            selected_index = sample_with_replacement(index, n_selected)
            total_selected_frames += n_selected
            # print("Sampled WITH replacement {0} frames".format(selected_index.shape[0]))
            
        if i>0:
            increment += phase_seq_counts[i-1]
            index_all.extend([x + increment for x in sorted(selected_index)])
        else:
            index_all.extend(sorted(selected_index))


    # print("{0} frames in the sampled video".format(total_selected_frames))

    writer = imageio.get_writer(out_video, fps=30, codec='mjpeg', quality=10, pixelformat='yuvj444p')

    for frame_id in tqdm(index_all):
        # for frame_id in tqdm(lst):
        img = video.get_data(frame_id)
        writer.append_data(img)
    
    writer.close()

    frame_ids = np.array(list(range(len(index_all))), dtype=np.int)
    phase_data_short = np.vstack((frame_ids, np.array(phase_data[index_all], dtype=np.int))).transpose()
    np.savetxt(out_phase, phase_data_short, delimiter=',', fmt='%d')


def join_video_for_pretraining(video_folder_loc, phase_path_list, out_video, out_phase):

    count = []
    for phase_path in phase_path_list:
        # print(phase_path)
        phase_data = pd.read_csv(phase_path, header=None).values[:,1]
        count.append(phase_data.shape[0])

    
    #writer = imageio.get_writer(out_video, fps=30, codec='mjpeg', quality=10)

    phase_data_joined = []
    video_list = []
    for phase_path in phase_path_list:
        _, file_name = os.path.split(phase_path)
        file_name, _ = os.path.splitext(file_name)

        video_path = os.path.join(video_folder_loc, file_name+'.avi')
        video_list.append(VideoFileClip(video_path))

        #video = imageio.get_reader(video_path)
        phase_data_joined.extend(list(pd.read_csv(phase_path, header=None).values[:,1]))

    phase_data_joined = np.array(phase_data_joined)
    frame_nr = np.arange(0, np.sum(count))

    phase_data_joined = np.vstack((frame_nr, phase_data_joined))

    video_combined = concatenate_videoclips(video_list)


    video_combined.write_videofile(out_video)
    np.savetxt(out_phase, phase_data_joined, delimiter=',', fmt='%d')




if __name__ == "__main__":


    video_folder_loc = os.path.join(sys.argv[1],"videos")
    phase_path_list = glob(os.path.join(sys.argv[1], 'phase_annotations', '*.csv'))
    out_base_path = sys.argv[2]

    # video_folder_loc = os.path.join("/media/avemuri/E130-Project/MICCAIChallenges/Endoviz2018/workflow_challenge/TrainingSet-1/test/","videos")
    # phase_path_list = glob(os.path.join("/media/avemuri/E130-Project/MICCAIChallenges/Endoviz2018/workflow_challenge/TrainingSet-1/test/", 'phase_annotations', '*.csv'))
    # out_base_path = "/media/avemuri/E130-Project/MICCAIChallenges/Endoviz2018/workflow_challenge/TrainingSet-1/test/cleaned/"

    postfix = "cleaned"
    if len(sys.argv) > 3:
        postfix = sys.argv[3]

    fraction = 0.1
    if len(sys.argv) > 4:
        fraction = float(sys.argv[4])


    create_folders_if_necessary(os.path.dirname(out_base_path))
    create_folders_if_necessary(os.path.join(out_base_path, 'videos'))
    create_folders_if_necessary(os.path.join(out_base_path, 'phase_annotations'))

    
    # if not os.path.exists(os.path.dirname(out_base_path)):
    #     os.makedirs(out_base_path)

    # if not os.path.exists(os.path.join(out_base_path, 'videos')):
    #     os.makedirs(os.path.join(out_base_path, 'videos'))

    # if not os.path.exists(os.path.join(out_base_path, 'phase_annotations')):
    #     os.makedirs(os.path.join(out_base_path, 'phase_annotations'))

    # video_folder_loc = glob('E:\\data\\Endoviz2018\\Workflow_Challenge\\TrainingSet-1\\videos\\*.avi')
    # video_folder_loc = '/media/avemuri/E130-Project/MICCAIChallenges/Endoviz2018/workflow_challenge/TrainingSet-1/videos/'
    # phase_path_list = glob('/media/avemuri/E130-Project/MICCAIChallenges/Endoviz2018/workflow_challenge/TrainingSet-1/other/phase_annotation/*.csv')
    # out_base_path = '/media/avemuri/E130-Project/MICCAIChallenges/Endoviz2018/workflow_challenge/TrainingSet-1/compressed_data/'

    # video_folder_loc = 'E:\\data\\Endoviz2018\\Workflow_Challenge\\TrainingSet-1\\videos\\'
    # phase_path_list = glob('E:\\data\\Endoviz2018\\Workflow_Challenge\\TrainingSet-1\\phase_annotations\\*.csv')
    # out_base_path = 'E:\\data\\Endoviz2018\\Workflow_Challenge\\TrainingSet-1\\compressed_data\\'

    # video_out_path = os.path.join(out_base_path, 'videos', postfix +'_joined.mp4')
    # phase_out_path = os.path.join(out_base_path, 'phase_annotations', postfix + '_joined.csv')
    # join_video_for_pretraining(video_folder_loc, phase_path_list, video_out_path, phase_out_path)
    
    # print(phase_path_list)
    for loc in tqdm(phase_path_list):
        _, file_name = os.path.split(loc)
        file_name, _ = os.path.splitext(file_name)

        video_path = os.path.join(video_folder_loc, file_name+'.avi')
        phase_path = loc
        video_out_path = os.path.join(out_base_path, 'videos', file_name+'_'+postfix+'.avi')
        phase_out_path = os.path.join(out_base_path, 'phase_annotations', file_name+'_'+postfix+'.csv')

        compress_video_for_pretraining(video_path, phase_path, video_out_path, phase_out_path, fraction)
        # remove_blue_frames(video_path, phase_path, video_out_path, phase_out_path)

    # video_path = "/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/TrainingSet-1/videos/Prokto1.avi"
    # phase_path = "/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/TrainingSet-1/phase_annotations/Prokto1.csv"
    # out_phase = "/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/TrainingSet-1/cleaned/phase_annotations/Prokto1_cleaned.csv"
    # out_video = "/home/avemuri/DEV/Data/Endoviz2018/workflow_challenge/TrainingSet-1/cleaned/videos/Prokto1_cleaned.mp4"
        
    # remove_blue_frames(video_path, phase_path, out_video, out_phase, 100)
        

