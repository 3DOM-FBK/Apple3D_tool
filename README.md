# Apple3D tool

This work presents a smartphone-based pipeline for inspecting apple trees in the field, automatically detecting fruits from videos and quantifying their size and number. The proposed approach is intended to facilitate and accelerate farmers’ and agronomists’ fieldwork, making apple measurements more objective and giving a more extended collection of apples measured in the field while estimating harvesting/apple-picking dates. In order to do that rapidly and automatically, we propose a pipeline that combines photogrammetry, [Mask R-CNN](https://github.com/matterport/Mask_RCNN) and geometric algorithms.

![Apple3D tool framework](/assets/APPLE3D_FRAMEWORK.png)


The smart farming pipeline consists in the following steps: 

1.	Data acquisition: using a smartphone, a video of an apple tree is recorded in order to acquire images of the apples from multiple positions and angles. Some permanent targets located over the plant are needed in order to calibrate the phone’s camera and scale the produced photogrammetric results

Video examples are available at the following link: https://drive.google.com/drive/folders/1MXeWELoG_QFqEgvU2r--VMcrnWZgyIS2?usp=sharing

2.	Frame extraction: keyframes are extracted from the video in order to process them using a photogrammetric method. Our tool employs a 2D feature-based approach that discards blurred and redundant frames unsuitable for the photogrammetric process

To extract the frames from a video use:

```bash
python Apple3D_tool/smart_frame_extraction.py -v <file name or full path of the video> --out <output path folder> -s <sharpness threshold> -m <min step between frames> -M <max step between frames>
```

3.	Apples segmentation/ 4. masks generation: a pre-trained neural network model is used for apple’s instance segmentation on keyframes. This is an example showing the use of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) in a real world application. We train a model to detect apples on a tree and we then use the generated masks to facilitate the generation of a dense point cloud of these apples. Starting from the pre-trained weights from the [COCO](https://cocodataset.org/) dataset, a total of about [700 apple instances](https://drive.google.com/drive/folders/13DtJs90koMDqSBHWGVMPKiMnODfly9mW?usp=sharing) has been manually labelled and used for an additional training in order to increase the neural network’s performance

5.	Image orientation: the extracted keyframes are used for photogrammetric reconstruction purposes, starting from camera pose estimation and sparse point cloud generation

6.	Dense image matching: using the previously created masks within an MVS (Multi-View Stereo) process

7.	Apple size measurement: size and number of fruits are derived by fitting spheres to the photogrammetric point cloud (using a RANSAC algorithm)

![Apple3D main steps](/assets/APPLE3D_STEPS.png)


## Apple segmentation and mask generation

Install [Mask RCNN](https://github.com/matterport/Mask_RCNN/)

Download our [pre-trained weights](https://drive.google.com/drive/folders/1OxWQpDw7nDMTmkihV0wB496Fi7JEPAV-)

Run the script:

```
python3 Apple3D_tool/detect.py <weights folder path> <input folder path> <output folder path>
```
