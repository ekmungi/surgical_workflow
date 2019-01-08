import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage
import imageio
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import hamming
from skimage import filters

def get_points(point_list, descriptor, frame_nr, kind):
    points = list()
    for i, point in enumerate(point_list):
        x, y = point.pt
        single_point = {"x": x, "y": y, "frame_nr": frame_nr, "kind": kind, "descriptor": descriptor[i,:]}
        points.append(single_point)
    return points

def descriptor_to_matrix(descriptor):
    output = list()
    for element in descriptor:
        output.append(element)
    output = np.array(output)
    return output

def normalize_img(img):
    min = np.min(img)
    max = np.max(img)
    normalized_image = (img - min) / (max-min)
    return normalized_image

def blur_img(img, K=1):
    sigma = 1
    blured_image = filters.gaussian(img, k*sigma)
    return normalize_img(blured_image)

def DoG(img):
    #img1 = blur_img(img, K=1)
    blured_img = blur_img(img, K=1.6)
    mean_squared_error = np.sum((normalize_img(img) - blured_img) ** 2) / np.sum(img.shape)
    return mean_squared_error

def calc_distance(descriptor1, descriptor2):
    values1 = descriptor_to_matrix(descriptor1.values)
    values2 = descriptor_to_matrix(descriptor2.values)
    ## todo make it in both directions
    result = cdist(values1, values2, 'hamming')
    # best_point = np.argsort(result,axis=0)
    nearest_points = np.sort(result[::-1], axis=0)
    nearest_point = nearest_points[0, :]
    nearest_point = np.sort(nearest_point[::-1], axis=0)
    n_points = int(nearest_point.shape[0] * 0.25)
    dist_to_frame = np.mean(nearest_point[0:n_points])
    return dist_to_frame


video_path = "/home/rosst/Documents/endovis2018/Prokto1.avi"
sliding_window_size=10
video = imageio.get_reader(video_path)
n_frames = video._get_meta_data(0)['nframes']

orb = cv2.ORB_create()
surf = cv2.xfeatures2d.SURF_create(400)

features = pd.DataFrame(columns=["frame_nr", "kind", "x", "y"])
frame_distances = np.zeros(sliding_window_size,)
previous_keyframe_nr = -1
keyframe_counter = 0

for i in range(0,n_frames):
    # set variables
    is_keyframe = False
    k = 4

    # get current frame
    current_frame = video.get_data(i)

    # find features
    kp = orb.detect(current_frame) # find keypoints
    kp_orb, des_orb = orb.compute(current_frame, kp) # compute the descriptors with ORB

    kp_surf, des_surf = surf.detectAndCompute(current_frame, mask=None) # surf

    orb_points = pd.DataFrame(get_points(kp_orb, descriptor=des_orb, frame_nr=i, kind="ORB"))
    surf_points = pd.DataFrame(get_points(kp_surf, descriptor=des_surf, frame_nr=i, kind="SURF"))
    #frames = pd.concat([frames, orb_points, surf_points])
    features = pd.concat([features, orb_points])

    # calc distance
    if i>0:
        features_current_frame = features[features['frame_nr'] == i]
        features_previous_frame = features[features['frame_nr'] == i - 1]

        dist_to_frame = calc_distance(features_previous_frame.descriptor, features_current_frame.descriptor)
        # append dist at the beginning and delete last element
        frame_distances = np.append(frame_distances[1:],[dist_to_frame])

    # find keyframes
    if(i > sliding_window_size):

        frame_distances_mean = np.mean(frame_distances)
        frame_distances_std = np.sum(np.power((frame_distances - frame_distances_mean), 2)) * 1 / sliding_window_size
        if frame_distances[-1] < (frame_distances_mean - k * frame_distances_std):
            is_keyframe = True
        if (frame_distances_mean + frame_distances_std)<=frame_distances[-1]:
            is_keyframe = True

        ## todo correct threshold
        # if keyframe is too close to previous one
        if is_keyframe & (previous_keyframe_nr>-1):
            previous_keyframe = features[features['frame_nr'] == previous_keyframe_nr]
            dist_to_frame = calc_distance(previous_keyframe.descriptor, features_current_frame.descriptor)
            threshold = 0.6
            if dist_to_frame < threshold:
                is_keyframe = False

        # if keyframe is too blury
        if is_keyframe:
            if DoG(current_frame) < 0.75:
                is_keyframe = False


        if is_keyframe:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(current_frame, "KeyframeNr: {}".format(keyframe_counter), (10, 200), font, 4, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('keyframe', current_frame)
            previous_keyframe_nr = i
            keyframe_counter +=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
            print("Dist to previous keyframe: {}".format(dist_to_frame))


        cv2.imshow('current_frame', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # draw only keypoints location,not size and orientation
    #test = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
    #test = cv2.drawKeypoints(img,kp, outImage=np.zeros(np.shape(img)))
    #cv2.imshow('frame', test)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

