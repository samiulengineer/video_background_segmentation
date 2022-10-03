import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt



def capBackgroud(directory):

    imgList = []

    for i in os.listdir(directory):
        if i.endswith(".jpg"):
            img = cv2.imread(directory+"/"+i)
            imgList.append(img)

    background = np.mean(imgList, axis=0).astype(dtype=np.uint8)
    return background


def saveImg(imgs, imgName):

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    titles = ['Orginal Image', 'Background Image', 'Background Subtraction',
              'Binary masking', 'Opening(2 times)', 'Opening(4 times)']
    axs = axs.flatten()
    for img, ax, ti in zip(imgs, axs, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(ti)
        ax.axis('off')

    fig.tight_layout()
    plt.savefig(imgName, bbox_inches='tight', dpi=800)
    plt.clf()   # clears an axis
    plt.cla()   # clears the entire current figure with all its axes, but leaves the window opened
    plt.close()  # closes a window


def subBackground(img, background, threshold=30):

    subBack = cv2.absdiff(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
    ret, bw_img = cv2.threshold(subBack, threshold, 255, cv2.THRESH_BINARY)

    return subBack, bw_img


def opening(img):

    kernelSizes = [(1, 1), (2, 2)]
    openImg = img
    # loop over the kernels sizes
    for kernelSize in kernelSizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        openImg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return openImg


def gaussian(img):

    image = cv2.GaussianBlur(img, (5, 5), 1)
    #gaussianImg = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

    return image


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir")
    args = parser.parse_args()
    args = vars(args)
    
    directory = args["dataset_dir"]  # Input directory
    
    background = capBackgroud(directory)    # Capturing Background Image using median method

    outDir = "/home/mdsamiul/github_project/video_background_segmentation/background_subtraction_cv/output" # Output directory
    
    # check and create output directory
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    for i in os.listdir(directory):
        if i.endswith(".jpg"):
            img = cv2.imread(directory+"/"+i)

            # Subtracting Background and get binary mask
            subBack, bw_img = subBackground(img, background)

            # Applying opening technique (erosion followed by dilation)
            openingImg = opening(bw_img)
            openingImg2 = opening(openingImg)

            imgName = outDir + "/" + i  # figure name
            
            # save figure in output directory
            saveImg([img, background, subBack, bw_img,
                    openingImg, openingImg2], imgName)
            