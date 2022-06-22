import json
from datetime import datetime


def addStatistique(url, response, fileName):
    try:
        with open('example.json', 'r') as myfile:
            data = myfile.read()

        # parse file
        obj = json.loads(data)
        obj["from_url"].append({
            "date":  datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "response": response,
            "fileName": fileName
        })

        with open('example.json', 'w') as outfile:
            json.dump(obj, outfile)

    except:
        pass


def getData():
    try:
        with open('example.json', 'r') as myfile:
            data = myfile.read()
        return data
    except:
        return "error"
