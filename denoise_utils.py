"""
@author: Kamaleshwar M

"""
import os
import pickle
import librosa
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

def load_or_create_pickle(pickle_path: str, compute_func, *args):

    if os.path.exists(pickle_path):
        print(f'Loading cached data from {pickle_path}')
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f'Computing data and creating cache at {pickle_path}')
        data = compute_func(*args)
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        return data

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:

    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

def compute_stft(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    return magnitude, phase

def compute_ibm(clean_mag: np.ndarray, noise_mag: np.ndarray) -> np.ndarray:

    return np.greater(clean_mag, noise_mag).astype(float)

def process_audio_directory(directory: str) -> List[np.ndarray]:

    magnitudes = []
    phases = []
    
    return magnitudes, phases

def load_paired_data(clean_dir: str, noisy_dir: str, cache_dir: str = "cache") -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    os.makedirs(cache_dir, exist_ok=True)
    clean_pickle = os.path.join(cache_dir, "clean_data.pkl")
    noisy_pickle = os.path.join(cache_dir, "noisy_data.pkl")
    masks_pickle = os.path.join(cache_dir, "masks.pkl")

    clean_mags, clean_phases = load_or_create_pickle(
        clean_pickle,
        process_audio_directory,
        clean_dir
    )
    
    noisy_mags, noisy_phases = load_or_create_pickle(
        noisy_pickle,
        process_audio_directory,
        noisy_dir
    )
    
    def compute_masks(clean_mags, noisy_mags):
        masks = []
        for clean_mag, noisy_mag in tqdm(zip(clean_mags, noisy_mags), 
                                       desc='Computing masks',
                                       total=len(clean_mags)):
            masks.append(compute_ibm(clean_mag, noisy_mag))
        return masks
    
    masks = load_or_create_pickle(
        masks_pickle,
        compute_masks,
        clean_mags,
        noisy_mags
    )
    
    return clean_mags, noisy_mags, masks

def reconstruct_audio(magnitude: np.ndarray, phase: np.ndarray, mask: np.ndarray) -> np.ndarray:

    masked_magnitude =
    stft = 
    return librosa.istft(stft, hop_length=, win_length=)