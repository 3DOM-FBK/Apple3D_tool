# APPLE3D TOOL

This work presents a smartphone-based pipeline for inspecting apple trees in the field, automatically detecting fruits from videos and quantifying their size and number. The proposed approach is intended to facilitate and accelerate farmers’ and agronomists’ fieldwork, making apple measurements more objective and giving a more extended collection of apples measured in the field while estimating harvesting/apple-picking dates. In order to do that rapidly and automatically, we propose a pipeline that combines photogrammetry, [Mask R-CNN](https://github.com/matterport/Mask_RCNN) and geometric algorithms.

![Apple3D tool framework](/assets/APPLE3D_FRAMEWORK.png)


The smart farming tool is implemented in the following steps: 

1.	Data acquisition: using a smartphone, a video of an apple tree is recorded trying to image apples from multiple positions and angles. Some permanent targets are needed to be located over the plant in order to calibrate the phone’s camera and scale the produced photogrammetric results.

Videos are available at the following link: https://drive.google.com/drive/folders/1MXeWELoG_QFqEgvU2r--VMcrnWZgyIS2?usp=sharing

2.	Frame extraction: keyframes are extracted from the video in order to process them using a photogrammetric method. Our tool employs a 2D feature-based approach that discards blurred and redundant frames unsuitable for the photogrammetric process. 

Apply smart frame tool on a video. Requires OpenCV: 

```bash
python ./smart_frame_extraction.py -v E://Dati_Eleonora/ANNO_4/DIGITAL_FARMING/01__DATA/V1/ --out E://Dati_Eleonora/ANNO_4/DIGITAL_FARMING/01__DATA/V1/ -s 14 -m 5 -M 15
```

Where s, m and M are editable parameters and refers to sharpness (s), minimum (m) and maximum (M) frames between two consecutive keyframe 

3.	Apple segmentation - 4. Mask generation: a pre-trained neural network model is used for apple’s instance segmentation on keyframes. This is an example showing the use of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) in a real application. We train the model to detect apples only, and then we use the generated masks to facilitate the generation of a dense point cloud of the apple frutis. Starting from the pre-trained weights, a total of about [700 apple instances](https://drive.google.com/drive/folders/13DtJs90koMDqSBHWGVMPKiMnODfly9mW?usp=sharing) has been manually labelled and used for an additional training in order to increase the neural network’s performance.

5.	Image orientation: the extracted keyframes are used for photogrammetric reconstruction purposes, starting from camera pose estimation and sparse point cloud generation.

6.	Dense image matching: using the previously created masks within an MVS (Multi-View Stereo) process

7.	Apple size measurement: sizes and number of fruits are derived by fitting spheres to the photogrammetric point cloud (using a RANSAC algorithm)

![Apple3D main steps](/assets/APPLE3D_STEPS.png)


## APPLE SEGMENTATION AND MASK GENERATION

From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_balloon.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download `balloon_dataset.zip`. Expand it such that it's in the path `mask_rcnn/datasets/balloon/`.



## Run Jupyter notebooks
Open the `inspect_balloon_data.ipynb` or `inspect_balloon_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the Apple model

Train a new model starting from pre-trained COCO weights
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
```

The code in `balloon.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
