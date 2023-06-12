import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from mix_sound import audio_to_numpy, convert_audio_to_spectrogram_matrix, convert_spectrogram_matrix_to_audio, data_scale, data_rescale
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import soundfile as sf
import time

sampling_factor = 2
sample_rate = 8000 * sampling_factor
frame_length = 16384
hop_length_frame = 16384
hop_length_frame_noise = 5000 * sampling_factor
min_duration = 1.0
dim_square_spec = 64 * sampling_factor * np.power(2, sampling_factor)
n_fft = int(dim_square_spec * 2 - 1)
hop_length_fft = int(frame_length / dim_square_spec)

# sampling_factor = 1
# sample_rate = 8000 * sampling_factor
# frame_length = 8192
# hop_length_frame = 8192
# hop_length_frame_noise = 5000 * sampling_factor
# min_duration = 1.0
# dim_square_spec = 256
# n_fft = 511
# hop_length_fft = 32

weights_path = (r'Weights')
models_path = (r'Models')
# name_model = '\your_trained_model_here'
clean_noise_path = (r'Test\sound')
clean_path = (r'Test\clean_voice')
prediction_path = (r'test\predict')
# prediction_name = r'yourfilename.wav'
volume_db = 20

min_scale = -1
max_scale = 1

clean_noise_list = os.listdir(clean_noise_path)
clean_list = os.listdir(clean_path)

# y, sr = librosa.load(os.path.join(clean_noise_path, 'originalfilename.wav'), sr = sample_rate)

plt.plot(y)
plt.xlabel('samples')
plt.ylabel('amplitude')
plt.show()
    
clean_noise = audio_to_numpy(clean_noise_path, clean_noise_list, sample_rate, frame_length, hop_length_frame, min_duration)
clean = audio_to_numpy(clean_path, clean_list, sample_rate, frame_length, hop_length_frame, min_duration)
print(clean_noise.shape)


m_db_clean_noise, m_pha_angle_clean_noise = convert_audio_to_spectrogram_matrix(clean_noise, dim_square_spec, n_fft, hop_length_fft)
m_db_clean, m_pha_angle_clean = convert_audio_to_spectrogram_matrix(clean, dim_square_spec, n_fft, hop_length_fft)

plt.figure()
librosa.display.specshow(m_db_clean_noise[0,:,:], x_axis='time', y_axis='linear', sr=sample_rate, hop_length=hop_length_fft)
plt.colorbar()
plt.show()

model = keras.models.load_model(models_path+name_model)

X_in = m_db_clean_noise
X_ou = m_db_clean
X_ou = X_in - X_ou
X_in = data_scale(X_in, min_scale, max_scale)

X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
X_pred = model.predict(X_in)
X_pred_rescaled = data_rescale(X_pred, min_scale, max_scale, np.amin(X_ou), np.amax(X_ou))

plt.figure()
librosa.display.specshow(X_pred_rescaled[0,:,:,0], x_axis='time', y_axis='linear', sr=sample_rate, hop_length=hop_length_fft)
plt.colorbar()
plt.show()

X_denoised = (m_db_clean_noise - X_pred_rescaled[:,:,:,0])
X_denoised = X_denoised - (np.amax(X_denoised))

plt.figure()
librosa.display.specshow(X_denoised[0,:,:], x_axis='time', y_axis='linear', sr=sample_rate, hop_length=hop_length_fft)
plt.colorbar()
plt.show()


audio_denoised = convert_spectrogram_matrix_to_audio(X_denoised, m_pha_angle_clean_noise, frame_length, hop_length_fft)

nb_sounds = audio_denoised.shape[0]
denoise_file = audio_denoised.reshape(1, nb_sounds * frame_length) * (20 * np.log10(volume_db))
sf.write(os.path.join(prediction_path, prediction_name), denoise_file[0, :], sample_rate)
 
y, sr = librosa.load(os.path.join(prediction_path, prediction_name), sr = sample_rate)
plt.plot(y)
plt.xlabel('samples')
plt.ylabel('amplitude')
plt.show()