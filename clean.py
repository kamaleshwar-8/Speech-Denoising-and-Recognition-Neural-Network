"""
@author: Kamaleshwar M

"""
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio
import librosa

def envelope(y, rate, threshold):
    # Envelop part

def downsample_mono(path, sr):
    try:
        wav, rate = librosa.load(path, sr=None)
        wav = wav.astype(np.float32)

        # downsampling part
        
        return sr, wav
    
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None, None

def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)

def check_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def apply_voice_augmentation(wav, sr):
    augmented = []
    
    # Original
    augmented.append(wav)
    
    # agumentation part
    
    return augmented

def split_wavs(src_root, dst_root, delta_time=, sr=, threshold=):
    dt = delta_time
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)

    for src_dir in classes:
        # spilting part

if __name__ == '__main__':
    split_wavs(src_root='denoised_data', dst_root='clean', delta_time=, sr=, threshold=)