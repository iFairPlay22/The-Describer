from flask import Flask, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from IADecode import IADecode
import urllib.request
from flask_cors import CORS
global decoder
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "images/"
ALLOWED_EXTENSIONS = {'png', 'jpeg',
                      'jpg', 'tiff', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/iadecode/from_file', methods=['POST'])
def from_file():
    # Valid Image format and save it to the server

    if 'file' not in request.files:
        return "No file in request", 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return "No file selected", 400

    if file and allowed_file(file.filename):
        filename = file.filename  # secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(path))
        prediction = decoder.getPrediction(path)
        os.remove(path)
        return {"message": prediction}, 200
    else:
        return "File not allowed", 400

    # Ask AI to decode image

    # Remove image


@app.route('/iadecode/from_url', methods=['POST'])
def from_url():
    payload = request.get_json()
    path = payload['file']
    print("eee")
    fileName = os.path.basename(path)
    extension = os.path.splitext(path)[1]
    validExtensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']
    print("tot")
    # if(extension not in validExtensions):
    #     return "File not allowed", 400

    # Download image
    savePath = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(path, savePath)

    # Ask AI to decode image
    # Remove image
    predicrtion = decoder.getPrediction(savePath)
    os.remove(savePath)
    return {"message": predicrtion}, 200


if __name__ == "__main__":
    #encoder_model, decoder_model, vocabulary, transform, device = IADecode.getEncoder_model()
    decoder = IADecode()
    app.run(use_reloader=True, debug=True)
