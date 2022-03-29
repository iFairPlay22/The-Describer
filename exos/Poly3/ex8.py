import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from percentagePlot import PercentagePlot

# 1 couche Linear
#	Qu1 : quel est le % de bonnes prédictions obtenu au lancement du programme , pourquoi ?
#   - Le % de bonnes prédictions obtenu au lancement du programme est inférieur à 15% (généralement entre 9 et 12%).
#   - En effet, on a une chance sur 10 de trouver la bonne catégories pour chaque image.

#	Qu2 : quel est le % de bonnes prédictions obtenu avec 1 couche Linear ?
#   - Le % de bonnes prédictions obtenu au lancement du programme est de 90%.

#	Qu3 : pourquoi le test_loader n’est pas découpé en batch ?
#   - Le test_loader n’est pas découpé en batch car le temps de calcul n'est pas très important pour les tests.
#   - En effet, nous n'avons que 10000 images de test et les calculs ne sont pas très couteux. L'optimisation
#   - est en fait primordiale pour les données d'apprentissage car le calcul de gradient est gourmand en terme
#   - de temps d'execution. Mais dans la cadre des tests, on peut se passer d'un batch.

#   Qu4 : pourquoi la couche Linear comporte-t-elle 784 entrées ?
#   - La couche Linear comporte 784 entrées car elle prends en entrée une image de dimension 28x28 = 784.
#   - En effet, le dataset MNIST propose des images de dimention normalisée.

#   Qu5 : pourquoi la couche Linear comporte-t-elle 10 sorties ?
#   - La couche Linear comporte-t-elle 10 sorties car elle envoie une prédiction de 0 à 9 (le nombre de catégories).

# 2 couches Linear
#   Qu6 : quelles sont les tailles des deux couches Linear ?
#   - La première couche comporte 784 entrées et 128 sorties (donc 784 neurones).
#   - La deuxième couche comporte 128 entrées et 10 sorties  (donc 128 neurones).

# 	Qu7 : quel est l’ordre de grandeur du nombre de poids utilisés dans ce réseau ?
#   - L'ordre de grandeur du nombre de poids utilisés dans ce réseau est de : 784x128 + 128x10.

#	Qu8 : quel est le % de bonnes prédictions obtenu avec 2 couches Linear ?
#   - On obtient un pourcentage de bonnes prédictions très intéréssant avec les 2 couches Linear (plus de 97,5%).

# 3 couches Linear
#   Qu9 : obtient-on un réel gain sur la qualité des prédictions ?
#  - On n'obtient pas de réel gain sur la qualité des prédictions avec 3 couches (à peine 0.15% de précision gagné).

# Fonction Softmax
#   Qu10 : pourquoi est il inutile de changer le code de la fonction TestOK ?
#   - La fonction TestOK fonctionne de manière indépendante au fonctionnement des couches du réseau et du calcul de l'erreur totale.
#   - En effet, la fonction TestOK compte jute le nombre de fois on le réseau a prédit la bonne catégorie
#   - (pour chaque image, on récupère l'index correspondant au score maximal prédit, et on le compare
#   - au résultat réel). On a donc pas besoin de calculer l'entropie de la prédiction.


# Le réseau de neurones
class Net(nn.Module):

    # Initialisation du réseau
    def __init__(self):
        super(Net, self).__init__()

        self.a = 784
        self.b = 128
        self.c = 64
        self.d = 10

        # Couche 1 :
        # - Entrée : 784 = 28 * 28 (nombre de pixels dans chaque image)
        # - Sortie : 128
        self.FC1 = nn.Linear(self.a, self.b)

        # Couche 2 :
        # - Entrée : 128
        # - Sortie : 64 (catégories de sortie différentes)
        self.FC2 = nn.Linear(self.b, self.c)

        # Couche 3 :
        # - Entrée : 64
        # - Sortie : 10 (catégories de sortie différentes)
        self.FC3 = nn.Linear(self.c, self.d)

    # Fonction de calcul de la sortie du réseau
    def forward(self, x):

        n = x.shape[0]
        x = x.reshape((n, self.a))

        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        x = F.relu(x)
        x = self.FC3(x)

        return x

    # Calcul de l'erreur totale
    def Loss(self, Scores, target):

        # nb = Scores.shape[0]
        # TRange = torch.arange(0, nb, dtype=torch.int64)
        # scores_cat_ideale = Scores[TRange, target]
        # scores_cat_ideale = scores_cat_ideale.reshape(nb, 1)
        # delta = 1
        # Scores = Scores + delta - scores_cat_ideale
        # x = F.relu(Scores)
        # err = torch.sum(x)
        # return err

        # Permet de transformer des scores par catégorie en probabilité
        # d'appartenances à chaque catégorie
        y = F.log_softmax(Scores, dim=1)

        # On calcule la dispersion des données. Plus les données sont dispersées,
        # et moins on est sur de notre résultat, donc plus l'erreur totale est
        # grande.
        err = F.nll_loss(y, target)

        return err

    # Calcul du nombre de catégories correctement classifiées
    def TestOK(self, Scores, target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
        eq = pred == target                        # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK

##############################################################################

# Entrainement du réseau
def TRAIN(args, model, train_loader, optimizer, epoch):

    # Pour chaque batch (sous ensemble des données d'entrainement)
    for batch_it, (data, target) in enumerate(train_loader):

        # On réinitialise le gradient
        optimizer.zero_grad()

        # On calcule les scores du réseau (prédictions)
        Scores = model.forward(data)

        # On calcule l'erreur totale
        loss = model.Loss(Scores, target)

        # On ajuste les poids du réseau
        loss.backward()
        optimizer.step()

# Test des performances du réseau
def TEST(model, test_loader):
    ErrTot = 0
    nbOK = 0
    nbImages = 0

    # On désactive le calcul du gradient lors des tests, car on ne
    # le calcule que pendant les sessions d'entrainement
    with torch.no_grad():

        # Pour chaque batch (sous ensemble des données de test)
        for data, target in test_loader:

            # On calcule les scores du réseau (prédictions)
            Scores = model.forward(data)

            # On calcule le nombre de lignes correctement prédites
            nbOK += model.TestOK(Scores, target)

            # On calcule l'erreur totale
            ErrTot += model.Loss(Scores, target)

            # On update le nombre d'images par rappor à la taille du batch
            nbImages += data.shape[0]

    # On calcule le nombre moyen de lignes correctement prédites
    pc_success = 100. * nbOK / nbImages
    print(f'\nTest set:   Accuracy: {nbOK}/{nbImages} ({pc_success:.2f}%)\n')

    return pc_success

##############################################################################


def main(batch_size):

    # Normalisation des couleurs (channels)
    # On uniformise les niveaux de proportion des couleurs dans l'image afin d'obtenir
    # une distribution statistique ayant une moyenne de 0 et une variance de 1
    moy, dev = 0.1307, 0.3081
    TRS = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(moy, dev)])

    # On charge les images du dataset d'images "MNIST"
    path = './data/ex8'
    TrainSet = datasets.MNIST(path, train=True,
                              download=True, transform=TRS)
    TestSet = datasets.MNIST(path, train=False,
                             download=True, transform=TRS)

    # On charge les données par paquets de taille batch_size
    train_loader = torch.utils.data.DataLoader(TrainSet, batch_size)
    test_loader = torch.utils.data.DataLoader(TestSet, len(TestSet))

    # On initialise le réseau de neurones
    model = Net()
    optimizer = torch.optim.Adam(model.parameters())

    # On va lancer 40 sessions d'entrainement / test afin de converger
    totalEpochs = 40
    percentagePlot = PercentagePlot("ex8", 0, totalEpochs, 90, 100)

    # On teste le réseau la première fois afin de voir les performances initiales    
    predictionSuccess = TEST(model, test_loader)

    # On lance les sessions d'entrainement
    for epoch in range(totalEpochs):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        # On entraine le réseau
        TRAIN(batch_size, model,  train_loader, optimizer, epoch)

        # On teste le réseau
        predictionSuccess = TEST(model,  test_loader)
        percentagePlot.addPoint(predictionSuccess)

    percentagePlot.show()
    
# Lancement du programme principal
main(batch_size=64)
