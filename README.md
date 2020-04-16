# Projet de Visord : Reconnaissance faciale

Arthur Lindoulsi <br>
Terence Le Huu Phuong <br>

## Application de reconnaissance faciale:
Executer la commande 
`python faces.py`
Le paramètre "mode" peut être changé si l'on veut utiliser un réseau entraîné plutôt que openCV

## Deep learning:
Le module model contient le modèle VGG initialisé de pytorch.
Le module train contient les fonctions relatives à l'entraînement.
Le module utils contient les fonctions relatives à l'algorithme cascade.
Pour lancer un train : télécharger les datasets FER et BSDS300 
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

Les fusionner dans un csv enregistré au nom de "dataset.csv" contenant deux colonnes "train" et "eval" contenant toutes les images de train et d'eval respectivement avec leur label (1 si image de FER 0 si image d BSDS).
Enfin, sur un notebook jupyter, executer 

`run train.py`

`main()`
