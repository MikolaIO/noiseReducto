import librosa
import numpy as np
import os

def audio_to_frame(data, frame_length, hop_length_frame):

    sequence_sample_length = data.shape[0]
    print(sequence_sample_length)
    
    data_list = [data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]
    
    data_array = np.vstack(data_list)
    print(data_array.shape)
    
    return data_array

def audio_to_numpy(audio_dir, list_files, sample_rate, frame_length, hop_length_frame, min_duration):
    
    sound_array = []
    
    for file in list_files:
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        if (total_duration >= min_duration):
            sound_array.append(audio_to_frame(y, frame_length, hop_length_frame))
        else:
            print('File is below minimum duration')
            
    return np.vstack(sound_array)
    
def blend_noise(clean, noise, nb_samples, frame_length):

    clean_part = np.zeros((nb_samples, frame_length))
    noise_part = np.zeros((nb_samples, frame_length))
    clean_noise = np.zeros((nb_samples, frame_length))
    
    for i in range(nb_samples):
    
        random_clean = np.random.randint(0, clean.shape[0])
        random_noise = np.random.randint(0, noise.shape[0])
        noise_amp = np.random.uniform(0.2, 0.9)
        clean_part[i, :] = clean[random_clean, :]
        noise_part[i, :] = noise[random_noise, :] * noise_amp 
        clean_noise[i, :] = clean_part[i, :] + noise_part[i, :]
        
    return clean_part, noise_part, clean_noise
    
def convert_audio_to_mag_pha(audio, n_fft, hop_length_fft):

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stft_mag, stft_pha = librosa.magphase(stft)
    stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    stft_pha_angle = np.angle(stft_pha)
    
    return stft_mag_db, stft_pha_angle
    
def convert_audio_to_spectrogram_matrix(audio, dim_square_spec, n_fft, hop_length_fft):
    
    nb_audio = audio.shape[0]
    
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_pha_angle = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    
    for i in range(nb_audio):
        m_mag_db[i, :, :], m_pha_angle[i, :, :] = convert_audio_to_mag_pha(audio[i], n_fft, hop_length_fft)
        
    return m_mag_db, m_pha_angle
    
def convert_mag_pha_to_audio(m_mag_db, m_pha_angle, frame_length, hop_length_fft):
    
    m_mag_amp_reverse = librosa.db_to_amplitude(m_mag_db, ref=1.0)
    stft_reverse = m_mag_amp_reverse * np.exp(1.j * m_pha_angle)
    audio_reverse = librosa.istft(stft_reverse, hop_length=hop_length_fft, length=frame_length)
    
    return audio_reverse
    
def convert_spectrogram_matrix_to_audio(m_mag_db, m_pha_angle, frame_length, hop_length_fft):
    
    sound_array = []
    
    nb_spectrograms = m_mag_db.shape[0]
    
    for i in range(nb_spectrograms):
        audio_reverse = convert_mag_pha_to_audio(m_mag_db[i], m_pha_angle[i], frame_length, hop_length_fft)
        sound_array.append(audio_reverse)
        
    return np.vstack(sound_array)
    
def data_scale(X, min_scale, max_scale):
    min_val = np.amin(X)
    max_val = np.amax(X)
    X_std = ((X - min_val) / (max_val - min_val))
    X_scaled = X_std * (max_scale - min_scale) + min_scale
    
    return X_scaled

def data_rescale(X_scaled, min_scale, max_scale, min_val, max_val):
    X_std = (X_scaled - min_scale) / (max_scale - min_scale) 
    X_rescaled = X_std * (max_val - min_val) + min_val
    
    return X_rescaled
