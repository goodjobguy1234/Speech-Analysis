from sys import path
from flask import Flask, render_template, request, redirect
import os
import tensorflow as tf
from tensorflow import keras
import random
from pydub import AudioSegment

from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route("/", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
        # get file from POST request and save it
        #wav
            save_filename_path = "test.wav"

            audio_file = AudioSegment.from_file(file)
            audio_file.export(save_filename_path, format="wav")
            print("Show me: ", audio_file)
           # dst = "test.wav"
           # save_filename = 'static/' + secure_filename(file.filename)
           # file_name = str(random.randint(0, 100000))
           # audio_file.save(file_name)

        
    return render_template("index.html")


#@app.route("/submit", methods=['POST'])
#def submit():
#    # html -> .py
#    if request.method == "POST":
#        name = request.form["filename"]

    # .py -> html
#    return render_template("submit.html", n=name)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
