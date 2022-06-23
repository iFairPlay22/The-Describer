import os
import time
import urllib.request
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp'}
CKPT_FILES_TOKENS = {"encoder": "LJwDPw", "decoder": "mFlRWR"}


def downloadFile(path, outputpath):
    # print("Downloading file from: " + path)
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(path, outputpath)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validFileAndFileName(path):
    extension = os.path.splitext(path)[1].split("?")[0]
    if extension == ".svg":
        return None

    # Download image
    fileName = str(round(time.time() * 1000))
    fileName = fileName.replace("-", "").replace(":", "").replace(" ", "_")
    if extension != "" and extension != None:
        fileName = fileName + extension
    else:
        fileName = fileName + ".jpg"
    return fileName
