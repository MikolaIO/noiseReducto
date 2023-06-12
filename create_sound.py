from mix_sound import audio_to_numpy, blend_noise, convert_audio_to_spectrogram_matrix
import os
import soundfile as sf
import numpy as np

clean_path = (r'Train\clean_voice')
noise_path = (r'Train\noise')
sound_path = (r'Train\sound')
spectrogram_path = (r'Train\spectrogram')
time_serie_path = (r'Train\time_serie')
weights_path = (r'Weights')
file_version = ('5000_1')
sampling_factor = 1
sample_rate = 8000 * sampling_factor
frame_length = 16384        # 16384 for 16khz, 8064 for 8khz
hop_length_frame = 16384    # 16384 for 16khz, 8064 for 8khz
hop_length_frame_noise = 5000 * sampling_factor
min_duration = 1.0
nb_samples = 5000
dim_square_spec = 128
n_fft = 1023
hop_length_fft = 32
list_clean = os.listdir(clean_path)
list_noise = os.listdir(noise_path)

clean = audio_to_numpy(clean_path, list_clean, sample_rate, frame_length, hop_length_frame, min_duration)
noise = audio_to_numpy(noise_path, list_noise, sample_rate, frame_length, hop_length_frame_noise, min_duration)

clean_part, noise_part, clean_noise = blend_noise(clean, noise, nb_samples, frame_length)
print(clean_part.shape)
print(noise_part.shape)
print(clean_noise.shape)

clean_noise_1d = clean_noise.reshape(1, nb_samples * frame_length)
sf.write(sound_path + '\clean_noise' + file_version + '.wav', clean_noise_1d[0, :], sample_rate)
clean_1d = clean_part.reshape(1, nb_samples * frame_length)
sf.write(sound_path + '\clean' + file_version + '.wav', clean_1d[0, :], sample_rate)
noise_1d = noise_part.reshape(1, nb_samples * frame_length)
sf.write(sound_path + r'\noise' + file_version + '.wav', noise_1d[0, :], sample_rate)

m_db_clean, m_pha_clean = convert_audio_to_spectrogram_matrix(clean_part, dim_square_spec, n_fft, hop_length_fft)
m_db_noise, m_pha_noise = convert_audio_to_spectrogram_matrix(noise_part, dim_square_spec, n_fft, hop_length_fft)
m_db_clean_noise, m_pha_clean_noise = convert_audio_to_spectrogram_matrix(clean_noise, dim_square_spec, n_fft, hop_length_fft)

np.save(time_serie_path + '\clean_timeserie' + file_version, clean_part)
np.save(time_serie_path + r'\noise_timeserie' + file_version, noise_part)
np.save(time_serie_path + '\clean_noise_timeserie' + file_version, clean_noise)


np.save(spectrogram_path + '\clean_db' + file_version, m_db_clean)
np.save(spectrogram_path + r'\noise_db' + file_version, m_db_noise)
np.save(spectrogram_path + '\clean_noise_db' + file_version, m_db_clean_noise)

np.save(spectrogram_path + '\clean_pha_db' + file_version, m_pha_clean)
np.save(spectrogram_path + r'\noise_pha_db' + file_version, m_pha_noise)
np.save(spectrogram_path + '\clean_noise_pha_db' + file_version, m_pha_clean_noise)








