from Models.IADecodeManager import IADecodeManager
from Services import AuthService, ImageService, StatistiqueService
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv, dotenv_values
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "images/"

global DecoderManager


@app.route('/statistique', methods=['GET'])
def getStat():
    return StatistiqueService.getData(), 200


@app.route('/iadecode/from_file/<lang>', methods=['POST'])
def from_file(lang):

    # Valid Image format and save it to the server
    if not AuthService.verify_token(request.headers):
        return "Forbidden", 403

    if 'file' not in request.files:
        return "No file in request", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file and ImageService.allowed_file(file.filename):

        path = os.path.join(app.config['UPLOAD_FOLDER'],  file.filename)
        file.save(os.path.join(path))
        prediction = DecoderManager.getPrediction(path, lang)

        try:
            os.remove(path)
        except:
            pass

        return {"message": prediction}, 200
    else:
        return "File not allowed", 400


@app.route('/iadecode/from_url/<lang>', methods=['POST'])
def from_url(lang):

    if not AuthService.verify_token(request.headers):
        return "Forbidden", 403

    payload = request.get_json()
    path = payload['file']

    fileName = ImageService.validFileAndFileName(path)
    if(fileName == None):
        return "SVG not supported", 400

    savePath = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
    ImageService.downloadFile(path, savePath)

    prediction = DecoderManager.getPrediction(savePath, lang)

    try:
        os.remove(savePath)
    except:
        pass

    return {"message": prediction}, 200


@app.after_request
def after_request_func(response):
    StatistiqueService.addStatistique(
        request.url, response.status_code)
    return response


if __name__ == "__main__":

    DecoderManager = IADecodeManager(4)
    app.run()
