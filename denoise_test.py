# test.py

import os
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
from tqdm import tqdm

def load_audio_file(file_path: str) -> tuple:


    return signal, sr

def compute_stft(signal: np.ndarray) -> tuple:

    return magnitude, phase

def reconstruct_audio(magnitude: np.ndarray, phase: np.ndarray, mask: np.ndarray) -> np.ndarray:
   
    return librosa.istft(stft, hop_length=, win_length=)

def process_magnitude_in_chunks(model, magnitude: np.ndarray, chunk_size: int = ) -> np.ndarray:

    
    for start_idx in range(0, time_steps, chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, time_steps)
        
        # Handle chunks
        if end_idx - start_idx < chunk_size:
            # Pad the chunk if needed
            
        else:
           
            
        # Prepare input - ensure it matches the expected shape
        chunk_input = chunk.T.reshape()
        chunk_input = chunk_input / ()
        
        # Get prediction
        try:
            chunk_mask = model.predict(, verbose=)
            chunk_mask = chunk_mask[0].T
            
            if is_final_chunk:
                # For final chunk, only take the relevant portion
                
            else:
                # For normal chunks, apply crossfade in overlap region
                if start_idx == 0:
                    
                else:
                    # Create crossfade weights
                    fade_in = 
                    fade_out = 
                    
                    # Apply crossfade
                    
                    
                    # Copy the non-overlapping part
                    if end_idx > overlap_end:
                       
        
        except ValueError as e:
            print(f"Error processing chunk: {str(e)}")
            continue
    
    return full_mask

def calculate_snr(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:

    return snr

def process_noisy_file(model, input_file: str, output_file: str) -> float:
    
            return snr
        
        return None
    
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return None

def batch_process_noisy_files(model_path: str, input_dir: str, output_dir: str):
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model
        print("Loading model...")
        model = load_model(model_path)
        
        # Get list of WAV files
        wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        
        if not wav_files:
            print(f"No WAV files found in {input_dir}")
            return
        
        # Process all WAV files with progress bar
        snr_values = []
        for filename in tqdm(wav_files, desc="Processing audio files"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"denoised_{filename}")
            
            snr = process_noisy_file(model, input_file, output_file)
            
            if snr is not None:
                snr_values.append(snr)
                tqdm.write(f"SNR for {filename}: {snr:.2f} dB")
        
        # Print summary statistics
        if snr_values:
            print("\nProcessing Complete!")
            print(f"Files processed: {len(snr_values)}/{len(wav_files)}")
            print(f"Average SNR: {np.mean(snr_values):.2f} dB")
            print(f"Min SNR: {np.min(snr_values):.2f} dB")
            print(f"Max SNR: {np.max(snr_values):.2f} dB")
            print(f"Std SNR: {np.std(snr_values):.2f} dB")
        else:
            print("\nNo files were successfully processed with SNR calculation.")
            
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")

def main():
    """Main function to run the script."""
    # Define paths
    MODEL_PATH = "models/speech_denoising_model.keras"
    INPUT_DIR = "denoise_test"
    OUTPUT_DIR = "denoised_test_output"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please run train.py first to train the model.")
        return
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found!")
        print("Please make sure the dataset structure is correct.")
        return
    
    # Process all files
    print(f"Starting batch processing...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    batch_process_noisy_files(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()