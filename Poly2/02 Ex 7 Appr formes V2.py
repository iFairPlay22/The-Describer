

import random
import math
# (x,y,category)
points= []
N = 30    # number of points per class
K = 3     # number of classes
for i in range(N):
   r = i / N
   for k in range(K):
      t = ( i * 4 / N) + (k * 4) + random.uniform(0,0.2)
      points.append( [ ( r*math.sin(t), r*math.cos(t) ) , k ] )




# On se propose de travailler avec 2 couches de neurones :
# Input => Linear => Relu => Linear => Scores





