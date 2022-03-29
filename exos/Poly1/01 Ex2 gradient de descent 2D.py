import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import math
random.seed(datetime.now())
# SPYDER 
# Tools > Preferences > IPython Console > Graphics > Backend 
# change it from "Inline/En ligne" to "Automatic".
 
cm = plt.cm.RdBu
 

#draw function g
def drawFunction(minn,maxx):
  plt.clf() 
  xx, yy = np.meshgrid(np.linspace(minn,maxx, 200),np.linspace(minn,maxx, 200))
  zz = g(xx,yy)
  axes = plt.gca()
  axes.set_ylim([minn,maxx])
  axes.set_xlim([minn,maxx])
  plt.contourf(xx,yy,zz, 200, cmap=plt.cm.rainbow)
  plt.colorbar() 
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.title("Gradient descent")


def g(x,y):
    return x * x + 0.4 * y * y + 0.2 * x - 0.2 * x * y + 0.1

drawFunction(-5,5)
 
color = [ "red", "blue", "yellow", "green", "white"]



for t in range(5): 
  angle = random.uniform(0,628) / 100
  x = 4 * math.cos( angle )
  y = 4 * math.sin( angle )
  
  k = 0.2                
  print("=============================================")
  # gradient descent algorithm 
  for i in range(30):
                
      # exercice : creez deux fonctions correspondant aux dérivées partielles de g(x,y)
      # appliquez la descente du gradient
      
      dgdx = 2 * x + 0.2 - 0.2 * y  ###
      dgdy = 0.4 * 2 * y - 0.2 * x    ###
         
      pas = 0.1                     ###
      x -= pas * dgdx               ###
      y -= pas * dgdy               ###
      
      print(g(x,y))        
    
      plt.scatter(x, y,  s=50, c= color[t] ,  marker='x')
  
      plt.pause(0.05) # 0.1 second between each iteration, usefull for nice animation
  
plt.show() # wait for windows close event

 