import os
import argparse
from metrics import *
from dataset import get_test_dataloader
from loss import *
from tensorflow import keras
from tensorflow.keras.models import load_model
from utils import show_predictions, get_config_yaml, create_paths, patch_show_predictions


# setup gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Parsing variable
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--load_model_name")
parser.add_argument("--plot_single", type=bool)
parser.add_argument("--index", type=int)
parser.add_argument("--experiment")
parser.add_argument("--patchify")
parser.add_argument("--patch_size")
args = parser.parse_args()

if args.plot_single == 'True':
    args.plot_single = True
else:
    args.plot_single = False

# Set up test configaration
# ----------------------------------------------------------------------------------------------
config = get_config_yaml('project/config.yaml', vars(args))
create_paths(config, test = True)

# Multiple GPU Setup
# ----------------------------------------------------------------------------------------------
# test strategy Muli-GPU or single-GPU
# strategy = set_gpu(config['gpu'])

# Dataset
# ----------------------------------------------------------------------------------------------
test_dataset = get_test_dataloader(config)

# Load Model
# ----------------------------------------------------------------------------------------------
print("Loading model {} from {}".format(
    config['load_model_name'], config['load_model_dir']))
# with strategy.scope(): # if multiple GPU is required
model = load_model(os.path.join(
    config['load_model_dir'], config['load_model_name']), compile=False)

# Prediction Plot
# ----------------------------------------------------------------------------------------------
print("Saving test predictions...")
if config['patchify']:
    print("call patch_show_predictions")
    patch_show_predictions(test_dataset, model, config)
else:
    show_predictions(test_dataset, model, config)

# Evaluation Score
# ----------------------------------------------------------------------------------------------
metrics = list(get_metrics(config).values())
adam = keras.optimizers.Adam(learning_rate=config['learning_rate'])
model.compile(optimizer=adam, loss=focal_loss(), metrics=metrics)
model.evaluate(test_dataset)
