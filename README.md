# video_background_subtraction

## Experiments

After setup the required folders and package run one of the following experiment. There are four experiments based on combination of parameters passing through `argparse` and `config.yaml`. Combination of each experiments given below.

When you run the following code based on different experiments, some new directories will be created;

1. csv_logger (save all evaluation result in csv format)
2. logs (tensorboard logger)
3. model (save model checkpoint)
4. prediction (validation and test prediction png format)

* **Comprehensive Full Resolution with Class Balance (CFR-CB)**: We balance the dataset biasness towards non-water class in this experiment.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 10 \
    --index -1 \
    --experiment cfr_cb \
    --patchify False \
    --height 240 \
    --width 320 \
    --weights True \
```

* **Patchify Half Resolution with Class Balance (PHR-CB)**: In this experiment we take a threshold value (19%) of water class and remove the patch images for each chip that are less than threshold value.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr_cb \
    --patchify True \
    --patch_size 256 \
    --weights False \
    --patch_class_balance True
```

## Testing

* **CFR-CB Experiment**

Run following model for evaluating train model on test dataset.

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --plot_single False \
    --index -1 \
    --patchify False \
    --height 240 \
    --width 320 \
    --experiment cfr_cb \
    --weights True \
```

* **PHR-CB Experiment**

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --plot_single False \
    --index -1 \
    --patchify True \
    --experiment phr_cb \
    --weights True \
```
