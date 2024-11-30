"""
@author: Kamaleshwar M

Email : mkamaleshwar80@gmail.com

"""
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class MelSpectrogram(layers.Layer):
    def __init__(self, sr=, n_mels=, n_fft=, hop_length=, **kwargs):

        super(MelSpectrogram, self).__init__(**kwargs)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def build(self, input_shape):
        
    
    def call(self, inputs):
        
        
        return tf.expand_dims(log_mel_spectrogram, axis=-1)
    
    def get_config(self):
        
        return config

def DNN_RNN(N_CLASSES=, SR=, DT=):
    
    return model

def LSTM(N_CLASSES=, SR=, DT=):
    
    return model

def Conv2D_LSTM(N_CLASSES=, SR=, DT=):

    return model

def TDNN(N_CLASSES=, SR=, DT=):

    return model

def TDNN_LSTM(N_CLASSES=, SR=, DT=):

    return model