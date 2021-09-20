from sys import path
from flask import Flask, render_template, request, redirect, send_file
import os
import tensorflow as tf
from tensorflow import keras
import random
from pydub import AudioSegment
import pathlib 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow_io as tfio

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from werkzeug.utils import secure_filename

AUTOTUNE = tf.data.AUTOTUNE
UPLOAD_FOLDER = 'uploads/'
data_dir = 'Speech-Analysis/mini_speech_commands'
commands = np.array(tf.io.gfile.listdir(str(data_dir)))

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  position = tfio.audio.trim(waveform, axis=0, epsilon=0.1)
  start = position[0]
  stop = position[1]
  waveform = waveform[start:stop]
  return waveform, label

def get_spectrogram(waveform):
  zero_padding = tf.zeros([20000] - tf.shape(waveform), dtype=tf.float32)

  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

@app.route("/", methods=["GET","POST"])
def uploads_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
        # get file from POST request and save it
        #wav
            mpath = pathlib.Path('Speech-Analysis\saved_model\my_model.h5')
            model = tf.keras.models.load_model(mpath)
            # audio_decode = decode_audio(file.read())

            sample_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(sample_file)

            # audio_numpy = audio_decode.numpy()
            # plt.rcParams["figure.figsize"] = [7.50, 3.50]
            # plt.rcParams["figure.autolayout"] = True
            # plt.plot(audio_numpy)
            # plt.show()
            # new_model.predict(audio_numpy)
            
            sample_file = pathlib.Path('Speech-Analysis/mini_speech_commands/fuck/fuck2.wav')
            sample_ds = preprocess_dataset([str(sample_file)])
            print(str(sample_file))
            tf.print(sample_ds)
          
            for spectrogram, label in sample_ds.batch(1):
              prediction = model(spectrogram)
              plt.bar(commands, tf.nn.softmax(prediction[0]))
              plt.title(f'Predictions for "{commands[label[0]]}"')
              plt.show()
        
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True, threaded=True)
