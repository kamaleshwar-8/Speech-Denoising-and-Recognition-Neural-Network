"""
@author: Kamaleshwar M

"""
import sys
import os
import traceback
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QMainWindow, 
                            QTextEdit, QProgressBar, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPalette, QBrush
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from model import MelSpectrogram
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from glob import glob
import pyaudio
import wave
import tempfile

class VoiceRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModels()
        self.initAudio()

    def initUI(self):
        self.setWindowTitle('Human Voice Recognition System')
        self.setGeometry(300, 300, 600, 800)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        header_label = QLabel('Voice Recognition System')
        header_label.setStyleSheet('font-size: 24px; font-weight: bold; margin: 10px;')
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        file_group = QWidget()
        file_layout = QHBoxLayout(file_group)
        self.file_label = QLabel('No file selected')
        self.file_button = QPushButton('Select Audio File')
        self.file_button.setStyleSheet('padding: 5px 15px;')
        self.file_button.clicked.connect(self.selectFile)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        layout.addWidget(file_group)

        record_group = QWidget()
        record_layout = QHBoxLayout(record_group)
        self.record_button = QPushButton('Start Recording')
        self.record_button.setStyleSheet('padding: 5px 15px;')
        self.record_button.clicked.connect(self.toggleRecording)
        self.record_label = QLabel('Not recording')
        record_layout.addWidget(self.record_label)
        record_layout.addWidget(self.record_button)
        layout.addWidget(record_group)

        model_group = QWidget()
        model_layout = QHBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(['All Models', 'LSTM', 'DNN_LSTM', 'Conv2D_LSTM', 'TDNN', 'TDNN_LSTM'])
        model_layout.addWidget(QLabel('Select Model:'))
        model_layout.addWidget(self.model_combo)
        layout.addWidget(model_group)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.predict_button = QPushButton('Analyze Voice')
        self.predict_button.setStyleSheet('''
            QPushButton {
                padding: 10px;
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        ''')
        self.predict_button.clicked.connect(self.predictVoice)
        layout.addWidget(self.predict_button)

        # Results display
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet('''
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                font-family: Arial;
                font-size: 20px;
            }
        ''')
        layout.addWidget(self.result_text)

    def loadModels(self):
        try:
            self.models = {}
            model_names = ['lstm', 'dnn_rnn', 'conv2d_lstm', 'tdnn', 'tdnn_lstm']
            for model_name in model_names:
                model_path = f'models/{model_name}.keras'
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(
                        model_path, 
                        custom_objects={'MelSpectrogram': MelSpectrogram}
                    )
                else:
                    print(f"Warning: Model {model_name} not found at {model_path}")
            
            if os.path.exists('clean'):
                self.classes = sorted(os.listdir('clean'))
                self.le = LabelEncoder()
                self.le.fit(self.classes)
            else:
                raise FileNotFoundError("'clean' directory not found")
                
        except Exception as e:
            self.result_text.setText(f"Error loading models: {str(e)}")
            self.predict_button.setEnabled(False)

    def initAudio(self):
        self.audio = pyaudio.PyAudio()
        self.stream =
        self.frames = 
        self.is_recording =
        self.record_seconds = 
        self.chunk = 
        self.sample_format = 
        self.channels = 
        self.fs = 

    def selectFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "WAV Files (*.wav);;All Files (*)",
            options=options
        )
        if fileName:
            self.selected_file = fileName
            self.file_label.setText(f'Selected: {os.path.basename(fileName)}')
            self.predict_button.setEnabled(True)

    def toggleRecording(self):
        if not self.is_recording:
            self.startRecording()
        else:
            self.stopRecording()

    def startRecording(self):
        self.is_recording = True
        self.record_button.setText('Stop Recording')
        self.record_label.setText('Recording...')
        self.frames = []

        self.stream = self.audio.open(format=self.sample_format,
                                      channels=self.channels,
                                      rate=self.fs,
                                      frames_per_buffer=self.chunk,
                                      input=True)

        QTimer.singleShot(self.record_seconds * 1000, self.stopRecording)

        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            QApplication.processEvents() 

    def stopRecording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.setText('Start Recording')
            self.record_label.setText('Recording stopped')

            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                wf = wave.open(tmpfile.name, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.sample_format))
                wf.setframerate(self.fs)
                wf.writeframes(b''.join(self.frames))
                wf.close()

            self.selected_file = tmpfile.name
            self.file_label.setText(f'Recorded: {os.path.basename(self.selected_file)}')
            self.predict_button.setEnabled(True)

    def process_magnitude_in_chunks(self, model, magnitude, chunk_size=):
        
        return full_mask

    def denoise_audio(self, input_file):

        try:

            denoising_model = load_model("models/speech_denoising_model.keras")
            
            return denoised_file
            
        except Exception as e:
            print(f"Error during denoising: {str(e)}")
            traceback.print_exc()
            return None

    def compute_stft(self, signal):


    def reconstruct_audio(self, magnitude, phase, mask):

    def predictVoice(self):
        if not hasattr(self, 'selected_file'):
            self.result_text.setText('Please select an audio file or record voice first')
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.predict_button.setEnabled(False)
        
        try:

            self.result_text.setText("Denoising audio...")
            self.progress.setValue(10)
            
            denoised_file = self.denoise_audio(self.selected_file)
            if not denoised_file:
                raise ValueError("Failed to denoise audio file")
            
            self.progress.setValue(30)
            
            sr =
            dt = 
            threshold = 

            self.result_text.setText("Processing denoised audio...")
            rate, wav = downsample_mono(denoised_file, sr)
            if rate is None:
                raise ValueError("Failed to load denoised audio file")
                
            mask, env = envelope(wav, rate, threshold=threshold)
            clean_wav = wav[mask]
            
            self.progress.setValue(50)
            
            step = 
            batch = []
            
            for i in range(0, clean_wav.shape[0], step):
            
                batch.append(sample)
            
            X_batch = np.array(batch, dtype=np.float32)
            
            selected_model = self.model_combo.currentText().lower()
            models_to_use = self.models if selected_model == 'all models' else {
                selected_model: self.models[selected_model]
            }

            self.result_text.setText("Analyzing voice...")
            results = {}
            predictions = []
        
            for idx, (model_name, model) in enumerate(models_to_use.items()):
                self.progress.setValue(60 + int((idx / len(models_to_use)) * 30))
            
                y_pred = model.predict(X_batch, verbose=0)
                y_mean = np.mean(y_pred, axis=0)
                predicted_class_index = np.argmax(y_mean)
                predicted_person = self.classes[predicted_class_index]
                confidence = float(y_mean[predicted_class_index])
                
                results[model_name] = {
                    'person': predicted_person,
                    'confidence': confidence
                }
                predictions.append(predicted_person)

            self.progress.setValue(100)

            result_text = "Analysis Results\n" # + "="*50 + "\n\n"
            # result_text += f"Denoised audio saved as: {os.path.basename(denoised_file)}\n\n"
            print(f"Denoised audio saved as: {os.path.basename(denoised_file)}\n\n")
            
            for model_name, result in results.items():
                # result_text += f"{model_name.upper()} Model:\n"
                # result_text += f"Predicted Person: {result['person']}\n"
                # result_text += f"Confidence: {result['confidence']:.2%}\n\n"
                print(f"{model_name.upper()} Model:\n")
                print(f"Predicted Person: {result['person']}\n")
                print(f"Confidence: {result['confidence']:.2%}\n\n")

            if len(models_to_use) == 1:
            
                result_text += f"{model_name.upper()} Model:\n"
                result_text += f"Predicted Person: {result['person']}\n"
                result_text += f"Confidence: {result['confidence']:.2%}\n\n"

            if len(models_to_use) > 1:
                from collections import Counter
                vote_results = Counter(predictions)
                majority_person, count = vote_results.most_common(1)[0]
                
                # result_text += "="*50 + "\n"
                result_text += "Final Decision:\n"
                result_text += f"Predicted Person: {majority_person}\n"
                result_text += f"Models in Agreement: {count}/{len(models_to_use)}\n"
                
                if count < len(models_to_use)/2 + 1:
                    result_text += "\nNote: Low agreement between models. Results may be unreliable."

            self.result_text.setText(result_text)
            
        except Exception as e:
            self.result_text.setText(f"Error during processing: {str(e)}")
        
        finally:
            self.progress.setVisible(False)
            self.predict_button.setEnabled(True)

            if self.selected_file.startswith(tempfile.gettempdir()):
                os.remove(self.selected_file)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = VoiceRecognition()
    ex.show()
    sys.exit(app.exec_())
