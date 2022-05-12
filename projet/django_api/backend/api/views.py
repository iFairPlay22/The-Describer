from rest_framework.response import Response
from rest_framework.decorators import api_view
from api.AI import decode
from rest_framework.exceptions import ParseError
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
import urllib.request


@api_view(['GET'])
def getFromFile(request):

    # Valid Image format and save it to the server
    file_obj = request.FILES['file']
    formatValid = ['image/png', 'image/jpeg',
                   'image/jpg', 'image/tiff', 'image/bmp', 'image/webp']
    if(file_obj.content_type not in formatValid):
        raise ParseError('File format not supported')

    path = default_storage.save(file_obj.name, ContentFile(file_obj.read()))
    tmp_file = os.path.join(settings.MEDIA_ROOT, path)

    # Ask AI to decode image
    response = Response({"message": decode.string(tmp_file)})

    # Remove image
    # os.remove(tmp_file)
    return response


@api_view(['GET'])
def getFromUrl(request):

    # Get image path and valid format
    path = request.data['path']
    fileName = os.path.basename(path)
    extension = os.path.splitext(path)[1]
    validExtensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']
    if(extension not in validExtensions):
        raise ParseError('File format not supported')

    # Download image
    savePath = settings.MEDIA_ROOT+fileName
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(path, savePath)

    # Ask AI to decode image
    response = Response({"message": decode.string(savePath)})

    # Remove image
    os.remove(savePath)
    return response
