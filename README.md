# **Video Background Subtraction**

## **Introduction**

Creating a foreground mask, which is a binary picture of the pixels that belong to moving elements in the scene, can be done with still cameras by using a technique known as background subtraction (BS), which is a common and commonly used method. In this study, we will show a video subtraction pipeline that makes use of both computer vision and deep learning techniques.

## **Dataset**

The dataset is CDNET can be downloaded from [here](http://jacarini.dinf.usherbrooke.ca/dataset2014). This dataset contains 11 video categories with 4 to 6 videos sequences in each category. However, we only used 8 video sequeces. They are: busStation, canoe, fountain02, highway, office, park, peopleInShade, sidewalk. Each individual video file (.zip or .7z) can be downloaded separately. Alternatively, all videos files within one category can be downloaded as a single .zip or .7z file. Each video file when uncompressed becomes a directory which contains the following:

1. a sub-directory named "input" containing a separate JPEG file for each frame of the input video
2. a sub-directory named "groundtruth" containing a separate BMP file for each frame of the groundtruth
3. "an empty folder named "results" for binary results (1 binary image per frame per video you have processed)
4. files named "ROI.bmp" and "ROI.jpg" showing the spatial region of interest
5. a file named "temporalROI.txt" containing two frame numbers. Only the frames in this range will be used to calculate your score

The groundtruth images contain 5 labels namely

- 0 : Static
- 50 : Hard shadow
- 85 : Outside region of interest
- 170 : Unknown motion (usually around moving objects, due to semi-transparency and motion blur)
- 255 : Motion

## **Environment Setup**

First clone the github repo in your local or server machine by following:

```
git clone https://github.com/samiulengineer/video_background_segmentation.git
```

Change the working directory to project root directory. Use Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```

Before start training check the variable inside config.yaml. Keep the above mention dataset in the data folder that give you following structure:

```
--data
    --busStation
        --groundtruth
        --input
    --canoe
        --groundtruth
        --input
    --fountain02
        --groundtruth
        --input
    --sidewalk
        --groundtruth
        --input
    --office
        --groundtruth
        --input
    --park
        --groundtruth
        --input
    --highway
        --groundtruth
        --input
    --peopleInShade
        --groundtruth
        --input    
```

------
------
------
_


# **Experiments (Computer Vision)**

After setting up the required folders and packages, run the following experiment. The experiment is based on a combination of parameters passing through `argparse`. There are eight folders in the data directory. You need to provide a path to any single folder's input directory.

When you run the following code, a new directory called output will be created. It will contain saved figures from the experiment.

```
python background_subtraction_cv/backround_subtraction_cv.py \
    --dataset_dir YOUR_DATASET_DIR/input
```

## **Results (Computer Vision)**

![Alternate text](/readme/bg_subplot.jpg)


----
----
----
_

# **Experiments (Deep Learning)**

After setup the required folders and package run one of the following experiment. There are two experiments based on combination of parameters passing through `argparse` and `config.yaml`. Combination of each experiments given below.

When you run the following code based on different experiments, some new directories will be created;

1. csv_logger (save all evaluation result in csv format)
2. logs (tensorboard logger)
3. model (save model checkpoint)
4. prediction (validation and test prediction png format)

## **Trained on a Single Dataset**

This experiment is for training the models on the frames collected from a single video. The dataset contains eight different folders. To run this experiment, you need to specify the folder name.

<FOLDER_NAME> = OR (fountain02, sidewalk, office, busStation, park, highway, canoe, peopleInShade)

```
python project/train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 10 \
    --index -1 \
    --experiment single_data \
    --height 240 \
    --width 320 \
    --single_dir <FOLDER_NAME>
```

## **Testing on a single dataset ("highway")**

Run following model for evaluating train model on test dataset.

```
python project/test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --plot_single False \
    --index -1 \
    --height 240 \
    --width 320 \
    --experiment single_data \
```

## **Prediction Result in "highway" dataeset (no.1000 frame)**

![Alternate text](/readme/1000frames.png)


--------------
--------------
--------------
_

# **Trained on a Multiple Dataset**

This experiment is for training the models on the whole dataset.

```
python project/train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 64 \
    --index -1 \
    --experiment multiple data \     
    --patchify True \
    --patch_size 240 
```



## **Test on a Multiple Dataset (8 Different Scenerios)**

```
python project/test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --plot_single False \
    --index -1 \
    --patchify True \
    --patch_size 240 \
    --experiment Multiple Dataset \ 
```


## **Prediction Result from 8 different Scenerios**

busStation dataset
![Alternate text](/readme/busStation.jpg)
canoe dataset
![Alternate text](/readme/canoe.png)
fountain02 dataset
![Alternate text](/readme/fountain02.png)
highway dataset
![Alternate text](/readme/highway.jpg)
office dataset
![Alternate text](/readme/office.jpg)
park dataset
![Alternate text](/readme/park.png)
peopleInShade dataset
![Alternate text](/readme/peopleInShade.jpg)
sidewalk dataset
![Alternate text](/readme/sidewalk.jpg)
