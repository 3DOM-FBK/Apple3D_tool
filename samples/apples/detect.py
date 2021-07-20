import sys, os
from mrcnn.config import Config
from mrcnn import visualize
import mrcnn.model as modellib
import skimage
from skimage.measure import find_contours
import numpy as np
import cv2
import json

class MeleConfig(Config):
    # Give the configuration a recognizable name
    NAME = "mele"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (mele)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1280

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 

class InferenceConfig(MeleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85

if __name__ == "__main__":

    model_directory = sys.argv[1]
    images_folder = sys.argv[2]
    output_folder = sys.argv[3]

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=model_directory)    

    print("Loading weights...")
    # model.load_weights(model.find_last(), by_name=True)
    model.load_weights(model.find_last(), by_name=True)

    image_paths = []

    segmentation = {}

    for filename in os.listdir(images_folder):

        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        
            image_path = os.path.join(images_folder, filename)

            img = skimage.io.imread(image_path)
            img_arr = np.array(img)
            results = model.detect([img_arr], verbose=1)
            r = results[0]
            
            # print(filename)

            masked_image = np.zeros(np.shape(img))
            #masked_image = img.astype(np.uint32).copy()

            regions = []            

            for roi in r['rois']:
                regions.append({"roi" : roi.tolist()})

            for i in range(len(r['rois'])):
                mask = r['masks'][:, :, i]
                for c in range(3):
                    masked_image[:, :, c] = np.where(mask == 1,
                                        255,
                                        masked_image[:, :, c])
                            
            cv2.imwrite(os.path.join(output_folder, filename), masked_image.astype(np.uint8))
        
            #visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], ['BG', 'mela'], r['scores'], figsize=(5,5))
            segmentation[filename] = regions            

    with open(os.path.join(output_folder, "segmentation.json"), 'w') as f:
        json.dump(segmentation, f, ensure_ascii=False)

