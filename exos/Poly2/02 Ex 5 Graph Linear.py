import torch, numpy, matplotlib.pyplot as plt

k = 50
n = 1
m = 3
o = 1

layer = torch.nn.Linear(n,m)
layer2 = torch.nn.Linear(m,o)	# creation de la couche Linear

activ = torch.nn.ReLU()         # fonction d’activation ReLU

Lx = numpy.linspace(-2,2,k)    # échantillonnage de 50 valeurs dans [-2,2]
Lx = Lx.reshape(k,n)

input = torch.FloatTensor(Lx)
v1 = layer(input)			        # utilisation du neurone
v2 = activ(v1)
v2 = layer2(v2)

Ly = v2.detach().numpy()		        # application de la fnt activation ReLU

# tracé
plt.plot(Lx,Ly) 	# dessine un ensemble de points
plt.axis('equal') 		# repère orthonormé
plt.show() 			    # ouvre la fenêtre d'affichage