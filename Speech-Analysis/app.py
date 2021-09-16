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
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)

  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

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
            new_model = tf.keras.models.load_model(mpath)

            audio_decode = decode_audio(file.read())
            audio_path = pathlib.Path(str(file))
            AUTOTUNE = tf.data.AUTOTUNE
            print(audio_path)
            audio_numpy = audio_decode.numpy()
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            plt.plot(audio_numpy)
            plt.show()
            # new_model.predict(audio_numpy)
        
        
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True, threaded=True)
