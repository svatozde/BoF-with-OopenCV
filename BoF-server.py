from flask import Flask, render_template, request, redirect, url_for, Response, send_file
import json
import os
from werkzeug.utils import secure_filename
import glob

import numpy as np
import cv2


ALLOWED_EXTENSIONS = ['jpg', 'jpeg']
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def index():
    return render_template("search.html", images=get_images_names())


@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


@app.route('/images/<string:name>', methods=['GET'])
def get_image(name=None):
    get_images_names()
    return send_file("uploads/"+name, mimetype='image/gif')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_images_names():
    image_names = glob.glob("./uploads/*")
    result = []
    for image_name in image_names:
        result.append(image_name.split("/")[-1])
    return result


if __name__ == '__main__':
    app.run()



