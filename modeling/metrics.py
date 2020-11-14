import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy, mean_absolute_error


def rounded_to_tenth_categorical_accuracy(y_true, y_pred):
    rounded_y_pred = tf.math.round(y_pred * 10) / 10
    # return K.mean(K.square(y_true - rounded_y_pred))
    return categorical_accuracy(y_true, rounded_y_pred)

def rounded_equal(y_true, y_pred):
    rounded_y_pred = tf.math.round(y_pred * 10) / 10
    # return K.mean(K.square(y_true - rounded_y_pred))
    return K.mean(K.equal(y_true, rounded_y_pred))

def rounded_mean_absolute_error(y_true, y_pred):
    rounded_y_pred = tf.math.round(y_pred * 10) / 10
    # return K.mean(K.square(y_true - rounded_y_pred))
    return mean_absolute_error(y_true, rounded_y_pred)