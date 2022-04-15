from turtle import color
import matplotlib.pyplot as plt
import time

class PercentagePlot:

    def __init__(self, title="percentage_plot", xStart=0, xEnd=10, yStart=0, yEnd=100):

        plt.figure()
        plt.xlim(xStart, xEnd)
        plt.ylim(yStart, yEnd)

        self.__data = {}
        self.__title = title

    def addPoint(self, name, color, score):

        if color not in self.__data:
            self.__data[color] = []
            self.__title += " " + color + "/" + str(name) + "%"

        self.__data[color].append(score)
        plt.scatter(len(self.__data[color]) - 1, score, color=color)

    def show(self):
        
        plt.title(self.__title)
        plt.show()

        plt.savefig('./images/' + str(time.time()) + "_" + self.__title + '.png')

