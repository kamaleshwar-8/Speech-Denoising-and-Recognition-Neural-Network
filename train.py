"""
@author: Kamaleshwar M

Email : mkamaleshwar80@gmail.com

"""
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from model import DNN_RNN, LSTM, Conv2D_LSTM, TDNN_LSTM, TDNN
from tqdm import tqdm
from glob import glob
import warnings
import librosa

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))
    
    def __getitem__(self, index):
        
        return X, Y
    
    def on_epoch_end(self):


def train_model(model, model_type, tg, vg, batch_size, sr, dt, epochs=100):
    logs_dir = 'logs'
    models_dir = 'models'
    for directory in [logs_dir, models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, f'{model_type}.keras'),
            monitor='val_loss',
            save_best_only=,
            save_weights_only=,
            mode=,
            save_freq=,
            verbose=
        ),
        CSVLogger(
            os.path.join(logs_dir, f'{model_type}_history.csv'),
            append=True
        ),
        # Early stopping
        # EarlyStopping(
        #     monitor='val_loss',
        #     patience=15,
        #     restore_best_weights=True,
        #     verbose=1
        # ),
        ReduceLROnPlateau(
            monitor=,
            factor=,
            patience=,
            min_lr=,
            verbose=
        )
    ]
    
    history = model.fit(
        tg,
        validation_data=vg,
        epochs=,
        verbose=,
        callbacks=callbacks
    )
    
    return history

def train(src_root, batch_size=, delta_time=, sample_rate=):
    sr = sample_rate
    dt = delta_time
    params = {
        'N_CLASSES': len(os.listdir(src_root)),
        'SR': sr,
        'DT': dt
    }
    
    models = {
        'dnn_rnn': DNN_RNN(**params),
        'lstm': LSTM(**params),
        'conv2d_lstm': Conv2D_LSTM(**params),
        'tdnn': TDNN(**params),
        'tdnn_lstm': TDNN_LSTM(**params)
    }
    
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(src_root))
    
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    
    wav_train, wav_val, label_train, label_val = train_test_split(
        wav_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    assert len(label_train) >= batch_size, 'Number of training samples must be >= batch_size'
    for split, labels in [('training', set(label_train)), ('validation', set(label_val))]:
        if len(labels) != params['N_CLASSES']:
            warnings.warn(
                f'Found {len(labels)}/{params["N_CLASSES"]} classes in {split} data. '
                'Increase data size or change random_state.'
            )
    
    tg = DataGenerator(
        wav_train, label_train, sr, dt,
        params['N_CLASSES'], batch_size=batch_size
    )
    vg = DataGenerator(
        wav_val, label_val, sr, dt,
        params['N_CLASSES'], batch_size=batch_size
    )
    
    results = {}
    for model_type, model in models.items():
        print(f"\nTraining {model_type} model...")
        history = train_model(model, model_type, tg, vg, batch_size, sr, dt)
        
        val_accuracy = max(history.history['val_accuracy'])
        results[model_type] = {
            'val_accuracy': val_accuracy,
            'history': history.history
        }
        print(f"{model_type} model - Best validation accuracy: {val_accuracy:.4f}")
    
    print("\nTraining completed. Final results:")
    for model_type, result in results.items():
        print(f"{model_type}: {result['val_accuracy']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['val_accuracy'])[0]
    print(f"\nBest performing model: {best_model}")

if __name__ == '__main__':
    train(src_root='clean', batch_size=, delta_time=,sample_rate=)