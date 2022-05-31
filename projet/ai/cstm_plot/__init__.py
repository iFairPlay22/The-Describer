import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class SmartPlot:

    def __init__(self, title="percentage_plot", x_label="x_label", y_label="y_label"):

        self.__data = {}
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label

    def addPoint(self, label, color, value):
        if label not in self.__data:
            self.__data[label] = { "color": color, "data": [] }

        self.__data[label]["data"].append(value)

    def build(self):
        self.__fig, self.__ax = plt.subplots()

        for label, k in self.__data.items():
            self.__ax.plot(range(len(k["data"])), k["data"], label=label, color=k["color"])

        self.__ax.set_xlabel(self.__x_label)
        self.__ax.set_ylabel(self.__y_label)
        self.__ax.set_title(self.__title)
        self.__ax.legend()

        if not os.path.exists("./output"):
            os.makedirs("./output")
        

        fileName = './output/plot_' + str(datetime.now())[0:19] + "_" + self.__title 
        fileName = fileName.replace("-", "_").replace(":", "_").replace(" ", "_")
        self.__fig.savefig(fileName + '.png')

    def show():
        plt.show()

if __name__ == "__main__":

    pp = SmartPlot("Scores", "Epochs", "Ratios")
    pp2 = SmartPlot("Gradients", "Epochs", "Loss")

    pp.addPoint("Ratio 1", "red", 0)
    pp.addPoint("Ratio 1", "red", 1)
    pp.addPoint("Ratio 1", "red", 2)
    pp.addPoint("Ratio 1", "red", 3)

    pp.addPoint("Ratio 2", "green", 1)
    pp.addPoint("Ratio 2", "green", 4)
    pp.addPoint("Ratio 2", "green", 4)
    pp.addPoint("Ratio 2", "green", 2)

    pp2.addPoint("Loss", "orange", 5)
    pp2.addPoint("Loss", "orange", 4)
    pp2.addPoint("Loss", "orange", 3)
    pp2.addPoint("Loss", "orange", 0)

    pp.build()
    pp2.build()

    SmartPlot.show()