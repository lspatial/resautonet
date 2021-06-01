
import numpy as np
import keras.backend as K

def r2K(y_true, y_pred):
    """
    rsquared for regression used in Keras
    :param y_true: array tensor for observation, probably multiple dimension.
    :param y_pred: array tensor for predictions, probably multiple dimension.
    :return: r2
    """
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2KAuto(y_true, y_pred):
    """
    rsquared for regression used in Keras
    :param y_true: array tensor for observation, just the last output .
    :param y_pred: array tensor for predictions, just the last output.
    :return: r2
    """
    SS_res =  K.sum(K.square(y_true[:,-1] - y_pred[:,-1]))
    SS_tot = K.sum(K.square(y_true[:,-1] - K.mean(y_true[:,-1])))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2np(y_true, y_pred):
    """
    rsquared for regression to process numpy's array type
    :param y_true: array tensor for observation, just the last output .
    :param y_pred: array tensor for predictions, just the last output.
    :return: r2
    """
    SS_res =  np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + 0.0000001) )

def mad(y_true, y_pred):
    """
    MAD for regression to process numpy's array type
    :param y_true: array tensor for observation, just the last output .
    :param y_pred: array tensor for predictions, just the last output.
    :return: r2
    """
    res =  np.mean(np.absolute(y_true - y_pred))
    return  res

def rmse2np(y_true, y_pred):
    """
    RMSE for regression to process numpy's array type
    :param y_true: array tensor for observation, just the last output .
    :param y_pred: array tensor for predictions, just the last output.
    :return: rmse
    """
    error=y_true-y_pred
    ret=np.sqrt(np.mean(np.square(error)))
    return ret
