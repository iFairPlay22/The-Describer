# import required module
import os
import time

def forEachFile(callback):

    # assign directory
    directory = '.'
    
    # iterate over files in that directory
    i = 1
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        
        # checking if it is a file
        if os.path.isfile(f) and f.endswith(".jpg"):
            callback(i, filename)
            i += 1

if __name__ == "__main__":

    forEachFile(lambda i, filename: os.rename(filename, str(time.time()).replace(".", "") + str(i) + ".jpg"))
    forEachFile(lambda i, filename: os.rename(filename, str(i) + ".jpg"))

    print("Ok")