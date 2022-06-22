import json
from datetime import datetime

FILEPATH = "./Assets/statistique.json"


def addStatistique(url, response):
    try:
        with open(FILEPATH, 'r') as myfile:
            data = myfile.read()

        # parse file
        obj = json.loads(data)
        if(url not in obj):
            print(obj)
            obj[url] = []
            print("toto")
        obj[url].append({
            "date":  datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "response": response,
        })

        with open(FILEPATH, 'w') as outfile:
            json.dump(obj, outfile)

    except Exception:
        print(Exception)
        pass


def getData():
    try:
        with open(FILEPATH, 'r') as myfile:
            data = myfile.read()
        return data
    except:
        return "error"
