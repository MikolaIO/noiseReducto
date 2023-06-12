import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from unet_model import unet
from autoencoder_model import autoencoder
from tensorflow import keras
from mix_sound import data_scale, data_rescale
import tensorflow as tf

spectrogram_path = (r'Train/spectrogram')
weights_path = (r'Weights')
models_path = (r'Models')
name_model = '/model_autoencoder_gpu'
file_version = '5000_1'
from_scratch = False
epochs = 25
batch_size = 20
min_scale = -1
max_scale = 1

X_in = np.load(spectrogram_path + r'/clean_noise_db' + file_version + '.npy')
X_ou = np.load(spectrogram_path + r'/clean_db' + file_version + '.npy')

X_ou = X_in - X_ou

X_in = data_scale(X_in, min_scale, max_scale)
X_ou = data_scale(X_ou, min_scale, max_scale)
 
print(np.amax(X_in),np.amin(X_ou))
     
X_in = X_in[:,:,:]
X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
X_ou = X_ou[:,:,:]
X_ou = X_ou.reshape(X_ou.shape[0],X_ou.shape[1],X_ou.shape[2],1)

X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)

if from_scratch:

    neural=unet()
    
else:

    neural = keras.models.load_model(models_path+name_model)


checkpoint = ModelCheckpoint(weights_path+name_model +'.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

history = neural.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[checkpoint], verbose=1, validation_data=(X_test, y_test))
neural.save(models_path+name_model)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.yscale('log')
plt.title('Training and validation loss')
plt.legend()
plt.show()

neural.summary()



    

    