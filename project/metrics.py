from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import segmentation_models as sm


# Keras MeanIoU
# ----------------------------------------------------------------------------------------------

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    '''
    Summary:
        MyMeanIOU inherit tf.keras.metrics.MeanIoU class and modifies update_state function.
        Computes the mean intersection over union metric.
        iou = true_positives / (true_positives + false_positives + false_negatives)
    Arguments:
        num_classes (int): The possible number of labels the prediction task can have
    Return:
        Class objects
    '''

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), sample_weight)


# Keras categorical accuracy
# ----------------------------------------------------------------------------------------------

def cat_acc(y_true, y_pred):
    '''
    Summary:
        This functions get the categorical accuracy
    Arguments:
        y_true (float32): list of true label
        y_pred (float32): list of predicted label
    Return:
        Categorical accuracy
    '''
    return keras.metrics.categorical_accuracy(y_true, y_pred)


# Custom dice coefficient metric
# ----------------------------------------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1):
    '''
    Summary:
        This functions get dice coefficient metric
    Arguments:
        y_true (float32): true label
        y_pred (float32): predicted label
        smooth (int): smoothness
    Return:
        dice coefficient metric
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_score(y_true, y_pred):
    return dice_coef(y_true, y_pred)


# Keras AUC metric
# ----------------------------------------------------------------------------------------------

def auc():
    return tf.keras.metrics.AUC(num_thresholds=3)


# Custom jaccard score
# ----------------------------------------------------------------------------------------------

def jaccard_score(y_true, y_pred, smooth=1):
    '''
    Summary:
        This functions get Jaccard score
    Arguments:
        y_true (float32): numpy.ndarray of true label
        y_pred (float32): numpy.ndarray of predicted label
        smooth (int): smoothness
    Return:
        Jaccard score
    '''
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (jac) * smooth + tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Custom
# ----------------------------------------------------------------------------------------------

# def evaluation_entry(fgim, gtim):

#     if (len(fgim.shape) == 3):
#         print("error: fgim mush be a gray image, fgim.shape:", fgim.shape)
#         return -1, -1, -1, -1

#     if (fgim.shape[0]*fgim.shape[1] != tf.math.reduce_sum(fgim == 0) + tf.math.reduce_sum(fgim == 2)):
#         print("error: fgim is not clean")
#         return -1, -1, -1, -1


#     TP = tf.math.reduce_sum((fgim == 2) & (gtim == 2))
#     FP = tf.math.reduce_sum((fgim == 2) & (gtim == 0))
#     TN = tf.math.reduce_sum((fgim == 0) & (gtim == 0))
#     FN = tf.math.reduce_sum((fgim == 0) & (gtim == 2))


#     return TP, FP, TN, FN

# def precision(y_true, y_pred):
#     TP, FP, _, _ = evaluation_entry(y_true, y_pred)
#     return TP / (TP + FP)

# def recall(y_true, y_pred):
#     TP, _, _, FN = evaluation_entry(y_true, y_pred)
#     return TP / (TP + FN)



# Metrics
# ----------------------------------------------------------------------------------------------

def get_metrics(config):
    """
    Summary:
        create keras MeanIoU object and all custom metrics dictornary
    Arguments:
        config (dict): configuration dictionary
    Return:
        metrics directories
    """

    m = MyMeanIOU(config['num_classes'])
    return {
        #'my_mean_iou': m,
        'f1_score': sm.metrics.f1_score,
        'precision': sm.metrics.precision,
        'recall': sm.metrics.recall,
        #'dice_coef_score': dice_coef_score
        # 'cat_acc':cat_acc # reduce mean_iou
    }
    
#metrics = ['acc']
