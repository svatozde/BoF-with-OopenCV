from flask import Flask, render_template, request, redirect, url_for, Response, send_file, send_from_directory
import json
import os
from werkzeug.utils import secure_filename
import glob
import cv2
from src.BAG import create_bag

# Config flask server
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "data/to_find/"
app.config['TRAIN_IMAGE'] = "data/samples/train/"

# Init bag
bag = create_bag()
sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)


@app.route('/', methods=['GET'])
def index():
    return render_template("search.html", images=[])


@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename), 0)

    key_points, descriptors = sift.detectAndCompute(img, None)
    similar_images = bag.find_similar_images(descriptors, 0.8)
    # bag.add_image(filename, descriptors)

    img_names = []
    for img_name in similar_images:
        img_names.append(img_name[0]['img_name'])

    return json.dumps(img_names), 200, {'ContentType': 'application/json'}


@app.route('/images/<string:name>', methods=['GET'])
def get_image(name=None):
    print('/images/'+name)
    return send_file(app.config['TRAIN_IMAGE'] + name, mimetype='image/gif')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
