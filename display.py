import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

result_path = (r'results')
result_list = os.listdir(result_path)
sample_rate = 8000
n_fft = 511
hop_length = int(n_fft / 2)

for file in result_list:
    y, sr = librosa.load(os.path.join(result_path, file), sr = sample_rate)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming')
    
    stft_mag = np.abs(stft)
    stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    
    plt.figure()
    librosa.display.specshow(stft_mag_db, x_axis='time', y_axis='linear', sr=sample_rate, hop_length=hop_length)
    plt.colorbar()
    plt.title(file)
    plt.show()