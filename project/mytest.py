import os

data_path = "/home/mdsamiul/github_project/video_background_subtraction/data/"
directories = os.listdir(data_path)

images = []
masks = []

for i in directories:
    fileDir = data_path + i 
    if os.path.isdir(fileDir) and i != "json":
        image_path = fileDir + "/input"
        mask_path = fileDir + "/groundtruth"
        image_names = os.listdir(image_path)
        image_names = sorted(image_names)
        mask_names = os.listdir(mask_path)
        mask_names = sorted(mask_names)
        
        print("{}:\n    input = {}    groundtruth = {}".format(i, len(image_names), len(mask_names)))