from flask import Flask, render_template, request, redirect, url_for, Response, send_file, send_from_directory
import json
import os
from werkzeug.utils import secure_filename
import glob
import pickle
import cv2

ALLOWED_EXTENSIONS = ['jpg', 'jpeg','png']
UPLOAD_FOLDER = "c:\\skola\\VMM\\jpg2\\"
sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
bag = None;
with open('BAG.pkl', 'rb') as f:
    bag = pickle.load(f)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def index():
    return render_template("search.html", images=[])


@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename), 0)

    key_points, descriptors = sift.detectAndCompute(img, None);
    similar_images = bag.getSimilar(descriptors, 0.98)
    bag.addImage(filename,descriptors)

    print(similar_images)

    img_names = []
    for img_name in similar_images:
        img_names.append(img_name[0].split("\\")[-1])


    return json.dumps(img_names), 200, {'ContentType': 'application/json'}


@app.route('/images/<string:name>', methods=['GET'])
def get_image(name=None):
    get_images_names()
    print('/images/<string:'+name+'>')
    return send_file(app.config['UPLOAD_FOLDER'] + "/" + name, mimetype='image/gif')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_images_names():
    image_names = glob.glob("./static/uploads/*")
    print(image_names)
    result = []
    for image_name in image_names:
        result.append(image_name.split("/")[-1])
    return result


if __name__ == '__main__':
    app.run(debug=True)
