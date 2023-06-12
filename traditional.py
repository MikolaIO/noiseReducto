import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

save_path = (r'Test\traditional')
save_name = r'\test.wav'
clean_noise_path = (r'Test\sound')
clean_noise_list = os.listdir(clean_noise_path)
sample_rate = 16000
n_fft = 1023
hop_length = int(n_fft / 2)

for file in clean_noise_list:
    y, sr = librosa.load(os.path.join(clean_noise_path, file), sr = sample_rate)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming')
    
    stft_mag = np.abs(stft)
    stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    print(stft_mag_db.shape)
    stft_pha = np.angle(stft)
    
    plt.figure()
    librosa.display.specshow(stft_mag_db, x_axis='time', y_axis='linear', sr=sample_rate, hop_length=hop_length)
    plt.colorbar()
    plt.show()
    
    stft_noise_mag = np.mean(stft_mag[:, :int(np.floor(stft_mag.shape[1] / 4))], axis=1)
    print(stft_noise_mag.shape)
    
    stft_noise_psd = stft_noise_mag ** 2
    
    stft_psd = stft_mag ** 2
    stft_clean_psd = stft_psd - stft_noise_psd[:, None]
    
    stft_clean_psd[stft_clean_psd < 0] = 0
    
    stft_clean_mag = np.sqrt(stft_clean_psd)
    
    stft_clean_mag_db = librosa.amplitude_to_db(stft_clean_mag, ref=np.max)
    
    plt.figure()
    librosa.display.specshow(stft_clean_mag_db, x_axis='time', y_axis='linear', sr=sample_rate, hop_length=hop_length)
    plt.colorbar()
    plt.show()
    
    stft_clean = stft_clean_mag * np.exp(1j * stft_pha)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, window='hamming')
    
    sf.write(save_path + save_name, audio_clean, sample_rate)