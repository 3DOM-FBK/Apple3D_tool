import numpy as np
import cv2 as cv
import argparse
import logging
import shutil
import glob
import time
import os   

logging.basicConfig(level=logging.INFO)

def select_keyframes(video_path, output_folder, buff_l, sharp_th, desc_dist_th, newness_min, newness_max, min_sep, max_sep):
    ''' ORB-based keyframe selection procedure used by Replicate. 

        Attributes:
            video_path (string)      :   path pf the video to be processed
            output_folder (string)              :   folder where the selected frames will be copied  
            buff_l (int)                        :   size of the ORB buffer
            sharp_th (float)                    :   sharpness threshold
            desc_dist_th (float)                :   mininum descriptor distance to consider a match good
            newness_min (float)                 :   _ (description todo)
            newness_max (float)                 :   _ (description todo)
            min_sep (int)                       :   Minimum frames between two consecutive keyframes
            max_sep (int)                       :   Maximum frames between two consecutive keyframe
    '''
    # ------------------------------------------ METHODS ----------------------------------------------
    def prepare_image(image):
        img = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        dim = (640, 360)
        # resize image
        resized = cv.resize(img, dim, interpolation = cv.INTER_CUBIC)
        return np.array(resized[:,:,0])

    def sharpness(img):
        H_g = np.array([                        # Gaussian filter std=1
            [0.0113,   0.0838,   0.0113],   
            [0.0838,   0.6193,   0.0838],
            [0.0113,   0.0838,   0.0113]])
        H_l = np.array([                        # Laplacian filter
            [0.1667,   0.6667,   0.1667],
            [0.6667,  -3.3333,   0.6667],
            [0.1667,   0.6667,   0.1667]])

        img_smooth = cv.filter2D(img, cv.CV_64FC1, H_g)                 # Apply Gaussian filtering
        img_lapl = cv.filter2D(img_smooth, cv.CV_64FC1, H_l)            # Apply Laplacian filter to detect edges
        img_lapl_smooth = cv.filter2D(img_lapl, cv.CV_64FC1, H_g)       # Apply Gaussian filter on the output of Laplacian filter
        return np.std(img_lapl - img_lapl_smooth)                       # Standard deviation between Laplacian and smoothed Laplacian
   
  
    logging.info('Selection parameters: \n\tVideo: {}\n\tOutput_folder: {}\n\tBuffer limit: {}\n\tSharpness threshold: {}\n\t' \
        'Descriptor min distance: {}\n\tNewness low threshold: {}\n\tNewness high threshold: {}\n\tMin frames between consecutive selections: {}\n\t' \
        'Max frames between consecutive selections: {}'.format(video_path, output_folder, buff_l, sharp_th,
        desc_dist_th, newness_min, newness_max, min_sep, max_sep))
    if not os.path.isdir(output_folder):
        logging.critical('Output folder "{}" does not exist'.format(output_folder))
        exit(1)

    keyframes = []                              # Selected keyframes
    desc_buff = None                            # Descriptor buffer
    curr_img_id = -1                            # Current frame id
    last_img_id = 0                             # Last selected frame id
    detector = cv.ORB_create(1000)              # ORB extractor
    matcher = cv.BFMatcher(cv.NORM_HAMMING)     # Feature matcher

    vidcap = cv.VideoCapture(video_path)
    success = True

    while success:
        success, image = vidcap.read()
		
        if not success:
            break
			
        curr_img_id += 1
        img_path = os.path.join(output_folder, "frame-%010d.jpg" % curr_img_id)
        img = prepare_image(image)
        logging.info('Processing frame %010d' % curr_img_id)       
        brightness = np.mean(img)		
        img = cv.equalizeHist(img)              

        S = sharpness(img)                            
        logging.info('Sharpness: {}'.format(S))       
        
        if brightness <= 50 or S < sharp_th: # Sharpness check
            continue

        if np.any(desc_buff) == None or curr_img_id - last_img_id > max_sep:        # Time selection
            _, desc_buff = detector.detectAndCompute(img, None)     
            cv.imwrite(img_path, image)
            keyframes.append(img_path)                           
            last_img_id = curr_img_id
            continue 

        _, desc = detector.detectAndCompute(img, None)                                             
        matches = matcher.match(desc, desc_buff)                                                  
        good_matches = [m.distance < desc_dist_th for m in matches]                               
        n_old_matches = sum(good_matches)                                           # True is 1 in the summation
        n_new_matches = desc.shape[0] - n_old_matches                                             

        newness = n_new_matches / float(n_old_matches)     
        if newness > newness_min and newness < newness_max and curr_img_id - last_img_id > min_sep:     # Descriptor selection
            cv.imwrite(img_path, image)
            keyframes.append(img_path)                      
            last_img_id = curr_img_id                                    
            
            new_idx = np.array([not x for x in good_matches])                           # Get the ids of the new keypoints (the not matched ones)
            new_desc = desc[new_idx, :]                                                 # Get the descriptors of those new keypoints 
            desc_buff = np.concatenate((new_desc, desc_buff))                           # Concatenate old descriptors with the new ones
            desc_buff = desc_buff[0:buff_l,:]                                           # TODO: check this Discard older descriptors according to the buffer limit

    logging.info('Selected keyframes: {}'.format(len(keyframes)))
    
    #for img_path in keyframes:                                                          # Copy the selected frames (high resolution) in the output folder
    #    img_name = os.path.split(img_path)[1]
    #    shutil.copy(os.path.join(high_res_frames_folder, img_name), os.path.join(output_folder, img_name))


def main():
    parser = argparse.ArgumentParser(description='Keyframe selection algorithm of the Replicate project')    
    parser.add_argument('-v', '--video',  help='Path to the video file', required=True)
    parser.add_argument('-out', '--output', help='Path where selected keyframe will be copied', required=True)

    parser.add_argument('-b', '--buff_limit', help='Size of the ORB features buffer', default=2000, type=int)
    parser.add_argument('-s', '--sharpness_th', help='Sharpness threshold', default=1.3, type=float)
    parser.add_argument('-d', '--descriptor_dist_th', help='Mininum descriptor differece to consider a match good', default=70, type=float)
    parser.add_argument('-nl', '--newness_low_th', help='_', default=0.05 / 0.95, type=float)
    parser.add_argument('-nh', '--newness_high_th', help='_', default=0.20 / 0.80, type=float)
    parser.add_argument('-m', '--min_n_frames', help='Minimum number of frames between two consecutive keyframes', default=10, type=int)
    parser.add_argument('-M', '--max_n_frames', help='Maximum number of frames between two consecutive keyframes', default=30, type=int)
    args = parser.parse_args()
    
    select_keyframes(args.video, args.output, args.buff_limit, args.sharpness_th, args.descriptor_dist_th, 
        args.newness_low_th, args.newness_high_th, args.min_n_frames, args.max_n_frames)


if __name__ == '__main__':
    main()