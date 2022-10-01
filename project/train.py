import os
import argparse
import time
from loss import *
from model import get_model, get_model_transfer_lr
from metrics import get_metrics
from tensorflow import keras
from utils import set_gpu, SelectCallbacks, get_config_yaml, create_paths
from dataset import get_train_val_dataloader
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

tf.config.optimizer.set_jit("True")
# mixed_precision.set_global_policy('mixed_float16')


# Parsing variable ctrl + /
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs")
parser.add_argument("--batch_size")
parser.add_argument("--index")
parser.add_argument("--experiment")
parser.add_argument("--patchify")
parser.add_argument("--patch_size")
parser.add_argument("--weights")
parser.add_argument("--patch_class_balance")

args = parser.parse_args()


# Set up train configaration
# ----------------------------------------------------------------------------------------------
config = get_config_yaml('project/config.yaml', vars(args))  # change by Rahat
create_paths(config)

# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Model = {}".format(config['model_name']))
print("Epochs = {}".format(config['epochs']))
print("Batch Size = {}".format(config['batch_size']))
print("Preprocessed Data = {}".format(os.path.exists(config['train_dir'])))
print("Class Weigth = {}".format(str(config['weights'])))
print("Experiment = {}".format(str(config['experiment'])))

# Dataset
# ----------------------------------------------------------------------------------------------
train_dataset, val_dataset = get_train_val_dataloader(config)


# Metrics
# ----------------------------------------------------------------------------------------------
metrics = list(get_metrics(config).values())  # [list] required for new model
# [dictionary] required for transfer learning & fine tuning
custom_obj = get_metrics(config)

# Optimizer
# ----------------------------------------------------------------------------------------------
learning_rate = 0.001
weight_decay = 0.0001
adam = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay)

# Loss Function
# ----------------------------------------------------------------------------------------------
#loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
custom_obj['loss'] = focal_loss()

# Compile
# ----------------------------------------------------------------------------------------------
# transfer learning
if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))) and config['transfer_lr']:
    print("Build model for transfer learning..")
    # load model and compile
    model = load_model(os.path.join(
        config['load_model_dir'], config['load_model_name']), custom_objects=custom_obj, compile=True)

    model = get_model_transfer_lr(model, config['num_classes'])
    model.compile(optimizer=adam, loss=loss, metrics=metrics)

# transfer learning
else:
    # fine-tuning
    if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))):
        print("Resume training from model checkpoint {}...".format(
            config['load_model_name']))
        # load model and compile
        model = load_model(os.path.join(
            config['load_model_dir'], config['load_model_name']), custom_objects=custom_obj, compile=True)

    # new model
    else:
        model = get_model(config)
        model.compile(optimizer=adam, loss=loss, metrics=metrics)

# Callbacks
# ----------------------------------------------------------------------------------------------
loggers = SelectCallbacks(val_dataset, model, config)
model.summary()

# Fit
# ----------------------------------------------------------------------------------------------
t0 = time.time()
history = model.fit(train_dataset,
                    verbose=1,
                    epochs=config['epochs'],
                    validation_data=val_dataset,
                    shuffle=False,
                    callbacks=loggers.get_callbacks(val_dataset, model),
                    )
print("training time minute: {}".format((time.time()-t0)/60))
