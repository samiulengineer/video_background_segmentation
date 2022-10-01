import os
from tensorflow.keras import backend as K
import tensorflow as tf
import segmentation_models as sm

# Setting framework

sm.set_framework('tf.keras')
sm.framework()

# setup gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def loss():
    '''
    Summary:
        This functions get the total loss calculating using dice loss and focal loss
    Arguments:
        None
    Return:
        Total loss (segmentation_models.base.objects.SumOfLosses object)
    '''
    weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
    dice_loss = sm.losses.DiceLoss(class_weights=weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    return total_loss

# keras.losses.custom_loss = loss()


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(
            logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    return loss


def bce_jaccard_loss(y_true, y_pred, smooth=1):
    '''
    Summary:
        This functions get Jaccard loss commonly referred to as the intersection-over-union loss
    Arguments:
        y_true (float32): numpy.ndarray of true label
        y_pred (float32): numpy.ndarray of predicted label
        smooth (int): smoothness
    Return:
        IOU loss or Jaccard loss
    '''
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth + tf.keras.losses.binary_crossentropy(y_true, y_pred)
