import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# 1 couche Linear
#	Qu1 : quel est le % de bonnes prédictions obtenu au lancement du programme , pourquoi ?
#   - Le % de bonnes prédictions obtenu au lancement du programme est inférieur à 15% (généralement entre 9 et 12%).

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


class Net(nn.Module):
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
        # - Sortie : 10 (catégories de sortie différentes)
        self.FC2 = nn.Linear(self.b, self.c)

        # Couche 3 :
        # - Entrée : 64
        # - Sortie : 10 (catégories de sortie différentes)
        self.FC3 = nn.Linear(self.c, self.d)

    def forward(self, x):

        n = x.shape[0]
        x = x.reshape((n, self.a))
        c1 = self.FC1(x)
        c1_relu = F.relu(c1)
        c1_relu_c2 = self.FC2(c1_relu)
        c1_relu_c2_relu = F.relu(c1_relu_c2)
        c1_relu_c2_relu_c3 = self.FC3(c1_relu_c2_relu)
        output = F.log_softmax(c1_relu_c2_relu_c3, dim=1)

        return output

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

        return F.nll_loss(Scores, target)

    def TestOK(self, Scores, target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
        eq = pred == target                      # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK

##############################################################################


def TRAIN(args, model, train_loader, optimizer, epoch):

    for batch_it, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        Scores = model.forward(data)
        loss = model.Loss(Scores, target)
        loss.backward()
        optimizer.step()

        # if batch_it % 50 == 0:
        #     print(
        #         f'   It: {batch_it:3}/{len(train_loader):3} --- Loss: {loss.item():.6f}')


def TEST(model, test_loader):
    ErrTot = 0
    nbOK = 0
    nbImages = 0

    with torch.no_grad():
        for data, target in test_loader:
            Scores = model.forward(data)
            nbOK += model.TestOK(Scores, target)
            ErrTot += model.Loss(Scores, target)
            nbImages += data.shape[0]

    pc_success = 100. * nbOK / nbImages
    print(f'\nTest set:   Accuracy: {nbOK}/{nbImages} ({pc_success:.2f}%)\n')

##############################################################################


def main(batch_size):

    # Chargement des données et normalisation des données
    moy, dev = 0.1307, 0.3081
    TRS = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(moy, dev)])
    TrainSet = datasets.MNIST('./data', train=True,
                              download=True, transform=TRS)
    TestSet = datasets.MNIST('./data', train=False,
                             download=True, transform=TRS)

    train_loader = torch.utils.data.DataLoader(TrainSet, batch_size)
    test_loader = torch.utils.data.DataLoader(TestSet, len(TestSet))

    # Création du réseau
    model = Net()
    optimizer = torch.optim.Adam(model.parameters())

    TEST(model,  test_loader)
    for epoch in range(40):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        TRAIN(batch_size, model,  train_loader, optimizer, epoch)
        TEST(model,  test_loader)


main(batch_size=64)
