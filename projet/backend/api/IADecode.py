import os


def string(path):
    return "hello world from " + os.path.relpath(path)