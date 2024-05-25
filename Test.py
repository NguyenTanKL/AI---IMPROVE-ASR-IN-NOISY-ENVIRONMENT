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
# Tải mô hình đã lưu
model = keras.models.load_model('CNN_model.h5')

# Tải danh sách test_files
test_files = np.load('test_files.npy', allow_pickle=True)
data_dir = pathlib.Path('SpeechCommands_Musan')
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Định nghĩa lại các hàm và preprocess_dataset từ file 1
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

label_name_x = np.array(tf.io.gfile.listdir(str(data_dir)))
first_subdir_path = data_dir / label_name_x[0]
label_name = np.array(tf.io.gfile.listdir(first_subdir_path))

def get_spectrogram(waveform):
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
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
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds

test_ds = preprocess_dataset(test_files).cache().prefetch(AUTOTUNE)

# Chuẩn bị dữ liệu kiểm tra
test_audio = []
test_labels = []
for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

# Chuyển đổi danh sách sang mảng NumPy

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

# Đảm bảo rằng test_audio có hình dạng đúng (số mẫu, chiều cao, chiều rộng, số kênh)
if test_audio.ndim == 3:
    test_audio = np.expand_dims(test_audio, -1)

# Thực hiện dự đoán
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

# Tính độ chính xác của bộ kiểm tra
test_acc = np.sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(14, 10))
sns.heatmap(confusion_mtx, xticklabels=label_name, yticklabels=label_name, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()