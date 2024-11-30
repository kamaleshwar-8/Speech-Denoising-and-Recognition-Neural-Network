# Speech Denoising and Recognition Neural Network

## Project Overview

This project, developed as part of the Machine Learning Systems course, presents an advanced neural network system for speech processing. It integrates two major components:
1. A deep learning-based speech denoising system
2. A sophisticated human speech recognition system

### Key Features

#### Speech Denoising
- Deep learning-based noise removal using Bidirectional LSTM networks
- Batch processing of audio files
- Signal-to-Noise Ratio (SNR) calculation
- Memory-efficient chunk-based processing
- Flexible handling of variable-length audio inputs

#### Speech Recognition
- Human voice recognition using ensemble neural network architectures
- Real-time voice recording and analysis support
- Multilingual and multi-speaker voice classification
- Ensemble model prediction with confidence scores

## Requirements

### System Requirements
- Python 3.8+

### Dependencies
- TensorFlow
- NumPy
- Librosa
- SoundFile
- tqdm
- PyQt5 (for GUI)

## Project Structure

### Speech Denoising
- `denoise_model.py`: Model architecture definition
- `denoise_train.py`: Training script
- `denoise_utils.py`: Utility functions for data processing
- `denoise_data.py`: Batch denoising script
- `denoise_test.py`: Testing and inference script

### Speech Recognition
- `model.py`: Model definitions for various neural networks
- `train.py`: Training script for speech recognition models
- `ui.py`: GUI application for real-time audio recording, analysis, and recognition


## Usage

### Speech Denoising

#### Training the Model
```bash
python denoise_train.py
```

#### Batch Denoising
```bash
python denoise_data.py
```

#### Testing
```bash
python denoise_test.py
```

### Speech Recognition

#### Training Models
```bash
python train.py
```

#### Running GUI Application
```bash
python ui.py
```

Select or record audio files using the provided GUI to analyze and view prediction results.

## Model Architectures

### Speech Denoising
- Bidirectional LSTM layers
- Layer Normalization
- Dropout for regularization
- Sigmoid activation for mask generation

### Speech Recognition
- Multiple architectures:
  - LSTM
  - TDNN
  - DNN-RNN
  - Conv2D-LSTM
  - TDNN-LSTM
- Mel Spectrogram feature integration for robust audio representation

## Performance Metrics

### Denoising Metrics
- Average Signal-to-Noise Ratio (SNR)
- Minimum and Maximum SNR
- SNR Standard Deviation

### Recognition Metrics
- Accuracy
- Confidence Scores

## Output



![1](https://github.com/user-attachments/assets/eba3be2c-c19f-4770-9b5c-a7a691a655dc)


![2](https://github.com/user-attachments/assets/1f53349d-e08a-451a-94f7-4cb575ab8670)


![3](https://github.com/user-attachments/assets/51935e6f-6011-4ff1-80ff-40faf3451610)


![4](https://github.com/user-attachments/assets/107faec6-e44f-4dd8-a99d-eeabf601ac4a)


## Acknowledgements

Special thanks to my friends Adhithya V, Akety Manjunath, Badri Prasanth G, HariHaran A M, Karthik Periyakarupphan M, Nidhish Kumar K and Velmugilan S, who contributed their voices to enhance the dataset for this project. Their diverse linguistic backgrounds and vocal characteristics were instrumental in developing a robust multilingual speech recognition model.

## Disclaimer

The entire code for this project cannot be shared as it includes certain components that I have fine-tuned and developed independently. For specific inquiries, feel free to reach out via email.

## Contact

**Author:** Kamaleshwar M  
**Email:** kamaleshwar.m.official@gmail.com
