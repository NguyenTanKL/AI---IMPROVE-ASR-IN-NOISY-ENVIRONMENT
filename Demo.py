import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from IPython import display
from pydub import AudioSegment
import io
from IPython import display
data_dir = pathlib.Path('SpeechCommands_Musan')
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*/*')
filenames = tf.random.shuffle(filenames)
print('Number of sample: ', len(filenames))
print('Example file tensor:', filenames[0])
train_files = filenames[:61627]
val_files = filenames[61628:69098]
test_files = filenames[69098:]
print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))
np.save('test_files.npy', test_files)
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_label(file_path):
 parts = tf.strings.split(file_path, os.path.sep)
 return parts[-2]

def decode_audio(audio_binary):
 audio, _ = tf.audio.decode_wav(audio_binary)
 return tf.squeeze(audio,axis=-1)

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    # convert_wav_to_pcm(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

label_name_x = np.array(tf.io.gfile.listdir(str(data_dir)))
first_subdir_path = data_dir / label_name_x[0]
label_name = np.array(tf.io.gfile.listdir(first_subdir_path))
def get_spectrogram(waveform):
 #Thực hiện zero_padding
 print(tf.shape(waveform))
 zero_padding = tf.zeros([16000] - tf.shape(waveform),dtype=tf.float32)
 waveform = tf.cast(waveform, tf.float32)
 equal_length = tf.concat([waveform, zero_padding], 0)
 #Thực hiện STFT
 spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=127)
 spectrogram = tf.abs(spectrogram)
 return spectrogram

def get_spectrogram_and_label_id(audio, label):
 spectrogram = get_spectrogram(audio)
 spectrogram = tf.expand_dims(spectrogram, -1)
 label_id = tf.argmax(label_name == label)
 return spectrogram, label_id

def preprocess_dataset(files):
 files_ds = tf.data.Dataset.from_tensor_slices(files)
 output_ds = files_ds.map(get_waveform_and_label,num_parallel_calls=AUTOTUNE)
 output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
 return output_ds

train_ds = preprocess_dataset(train_files)
val_ds = preprocess_dataset(val_files)

for item in train_ds.take(1):
 print(item)
batch_size = 256
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
for spectrogram, _ in train_ds.take(1):
 input_shape = spectrogram.shape
 print('Input shape:', input_shape)

num_labels = len(label_name)
normalizer=layers.Normalization()
normalizer.adapt(train_ds.map(lambda x, _: x))
print(num_labels)
model = models.Sequential([
layers.Input(shape=(124,129,1)),
layers.Resizing(64,64),
normalizer,
layers.Conv2D(64, 3, activation='relu'), 
layers.Conv2D(32, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Dropout(0.25),
layers.Flatten(),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(num_labels, activation = 'softmax'),
])
model.summary()
model.compile(
optimizer=tf.keras.optimizers.Adam(),
loss=tf.keras.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'],
)
EPOCHS = 50
historys = model.fit(
train_ds,
validation_data=val_ds,
epochs=EPOCHS,
callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=3),
)
model.save('CNN_model.h5')


