from flask import Flask, request
import os
import os.path
from os import path
import time
from IADecode import IADecode
import urllib.request
from flask_cors import CORS
from queue import Queue
from dotenv import load_dotenv, dotenv_values
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = "images/"
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp'}
CKPT_FILES_TOKENS = {"encoder": "LJwDPw", "decoder": "mFlRWR"}
decoders = [None]*4
decodersUsed = [False]*4


def downloadFile(path, outputpath):
    # print("Downloading file from: " + path)
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(path, outputpath)
    # print("Downloaded to: " + outputpath)


# if not(path.exists("./models_dir")):
#     os.mkdir("./models_dir")

# if not(path.exists("./models_dir/encoder.ckpt")):
#     downloadFile("https://transfer.sh/" +
#                  CKPT_FILES_TOKENS["encoder"] + "/encoder.ckpt", "./models_dir/encoder.ckpt")

# if not(path.exists("./models_dir/decoder.ckpt")):
#     downloadFile("https://transfer.sh/" +
#                  CKPT_FILES_TOKENS["decoder"] + "/decoder.ckpt", "./models_dir/decoder.ckpt")


def checkToken(playload):
    if "token" in playload:
        config = dotenv_values(".env.local")

        if playload["token"] == config["TOKEN"]:
            return True
    return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def loadDecoder():
    global decoders

    for i in range(len(decoders)):
        decoders[i] = IADecode()


def selectDecoder():
    global decodersUsed
    for i in range(len(decodersUsed)):
        if not decodersUsed[i]:
            decodersUsed[i] = True
            return (decoders[i], i)
    return None, -1


def freeDecoder(i):
    global decodersUsed
    decodersUsed[i] = False


@app.route('/iadecode/from_file/<lang>', methods=['POST'])
def from_file(lang):

    # Valid Image format and save it to the server
    decoder = None
    while(decoder == None):
        decoder, decoder_id = selectDecoder()
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
        prediction = decoder.getPrediction(path, lang)
        freeDecoder(decoder_id)
        try:
            os.remove(path)
        except:
            pass
        return {"message": prediction}, 200
    else:
        return "File not allowed", 400

    # Ask AI to decode image

    # Remove image


@app.route('/iadecode/from_url/<lang>', methods=['POST'])
def from_url(lang):

    decoder = None
    while(decoder == None):
        decoder, decoder_id = selectDecoder()

    payload = request.get_json()

    path = payload['file']

    extension = os.path.splitext(path)[1].split("?")[0]
    if extension == ".svg":
        return "SVG not supported", 400

    # Download image
    fileName = str(round(time.time() * 1000))
    fileName = fileName.replace("-", "").replace(":", "").replace(" ", "_")
    if extension != "" and extension != None:
        fileName = fileName + extension
    else:
        fileName = fileName + ".jpg"

    savePath = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
    downloadFile(path, savePath)

    # Ask AI to decode image
    # Remove image
    prediction = decoder.getPrediction(savePath, lang)
    freeDecoder(decoder_id)
    try:
        os.remove(savePath)
    except:
        pass
    return {"message": prediction}, 200


if __name__ == "__main__":

    loadDecoder()
    app.run()
