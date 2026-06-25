# Module IAApprentissage – Intelligence Artificielle

## 1. Présentation du module

Le module `IAApprentissage` regroupe la partie apprentissage automatique du projet **SocialAnalyzer**.

Son rôle est d'entraîner des modèles capables de prédire automatiquement, à partir d'un texte de tweet :
- une **polarité**
- un ou plusieurs **tons**
- une **cible thématique**

Ce module correspond à la partie **Intelligence Artificielle / Apprentissage** du projet. Il utilise une représentation TF-IDF des textes puis des réseaux de neurones construits avec Keras/TensorFlow.

## 2. Objectifs du module

Le module a plusieurs objectifs :
1. Transformer les textes en vecteurs numériques exploitables
2. Entraîner des modèles supervisés à partir du corpus annoté automatiquement
3. Utiliser Keras/TensorFlow pour respecter l'objectif du module d'IA
4. Gérer 3 tâches de classification :
   - polarité
   - ton
   - cible
5. Générer des prédictions sur un corpus non annoté simulant une API
6. Sauvegarder les modèles, les encodeurs, les prédictions et les statistiques

## 3. Place du module dans le pipeline

Le pipeline général du projet peut être résumé ainsi :
- Corpus brut
- normalisation / formatage
- weak supervision
- TF-IDF
- entraînement Keras
- prédiction sur corpus API
- comparaison avec gold standard

Le module `IAApprentissage` intervient principalement dans les étapes suivantes :
- TF-IDF
- entraînement des modèles
- prédiction

## 4. Architecture du module

Le module contient principalement les fichiers suivants :

```
Modules/IAApprentissage/ML/
- config.py
- trainTFIDF.py
- trainTensorLabel.py
- predictionTensor.py
- TrainedModels/
- README.md
```

Le dossier `TrainedModels/` contient les modèles et objets sauvegardés après l'entraînement.

## 5. Fichier `config.py`

Le fichier `config.py` centralise les chemins et paramètres utilisés par le module.

Il définit notamment :
- les chemins vers les corpus
- le chemin du modèle TF-IDF
- le dossier de sauvegarde des modèles
- le chemin du fichier de prédictions
- les paramètres du TF-IDF

Paramètres TF-IDF utilisés :

```
TFIDFMAXFEATURES = 20000
TFIDFNGRAMRANGE = (1, 4)
TFIDFMINDF = 3
TFIDFMAXDF = 0.95
TFIDFANALYZER = "char_wb"
```

Le TF-IDF est donc limité à 20 000 caractéristiques, ce qui permet de réduire la taille des vecteurs et d'éviter une surcharge mémoire trop importante.

## 6. Fichier `trainTFIDF.py`

Le fichier `trainTFIDF.py` entraîne le vectoriseur TF-IDF.

Il utilise le corpus : `Data/Corpus/CorpusFrenchTweets1-5M.csv`

Ce corpus massif de tweets francophones permet d'apprendre une représentation générale du langage utilisé sur Twitter/X.

Le script vérifie d'abord que le corpus existe et que la colonne `text` est bien présente. Il charge ensuite les textes, entraîne un `TfidfVectorizer` puis sauvegarde le modèle avec `joblib`.

Le modèle sauvegardé est : `Modules/IAApprentissage/ML/TrainedModels/tfidf.joblib`

## 7. Rôle du TF-IDF

Les modèles Keras ne peuvent pas travailler directement sur du texte brut. Il faut d'abord convertir les textes en données numériques.

Le TF-IDF sert donc à transformer chaque tweet en vecteur numérique, cela donne plus d'importance aux termes caractéristiques et réduit l'importance des mots trop fréquents.

Dans ce projet, le TF-IDF sert de représentation textuelle commune pour les 3 modèles.

## 8. Fichier `trainTensorLabel.py`

Le fichier `trainTensorLabel.py` entraîne les 3 modèles principaux du projet.

Il charge : `Data/Corpus/LabeledFrenchTweetPolitics.csv`

Ce fichier correspond au corpus politique annoté automatiquement par weak supervision.

Le script utilise ensuite le TF-IDF déjà entraîné pour transformer les textes en vecteurs numériques.

3 modèles Keras sont entraînés :
- polariteModelTensor.keras
- tonModelTensor.keras
- targetModelTensor.keras

## 9. Les trois tâches d'apprentissage

### 9.1 Polarité

La polarité est une classification multi-classe.

Labels possibles :

```
Favorable
Neutre
Défavorable
```

Une seule polarité est prédite pour chaque tweet.


### 9.2 Ton

Le ton est une classification multi-label.

Labels possibles :

```
Argumentatif
Autres
Informatif
Ironique
Question
```

Un tweet peut avoir plusieurs tons en même temps. Par exemple :

```text
Question;Informatif
```

### 9.3 Cible

La cible est une classification multi-classe.

Labels possibles :

```
Autre
Education
International
Politique
Sécurité
```

Une seule cible principale est prédite pour chaque tweet.

## 10. Architecture des modèles Keras

Les modèles sont des réseaux de neurones denses de type MLP.

Architecture générale :

```
Input
- Dense(512, relu)
- Dropout(0.2)
- Dense(256, relu)
- Dropout(0.2)
- Couche de sortie
```

La couche de sortie dépend de la tâche :

- `softmax` pour la polarité
- `sigmoid` pour le ton
- `softmax` pour la cible

Le `Dropout(0.2)` sert à réduire le risque de surapprentissage en désactivant aléatoirement une partie des neurones pendant l'entraînement.

## 11. Early stopping

Le script utilise un `EarlyStopping`.

Son rôle est d'arrêter l'entraînement si la perte de validation ne s'améliore plus pendant plusieurs époques.

Cela permet d'éviter d'entraîner inutilement le modèle et de limiter le surapprentissage.

## 12. Préparation des labels

### 12.1 Labels simples

Pour la polarité et la cible, les labels sont transformés en entiers avec `LabelEncoder`.

Exemple :

```
Neutre = 2
Favorable = 1
Défavorable = 0
```

Les encodeurs sont sauvegardés pour pouvoir décoder les prédictions plus tard.

### 12.2 Labels multi-label

Pour le ton, le script utilise `MultiLabelBinarizer`.

Exemple : `Question;Informatif` devient un vecteur binaire du type : `[0, 0, 1, 0, 1]`

Cela permet au modèle d'apprendre séparément la présence ou l'absence de chaque ton.

## 13. Sauvegarde des modèles

Après l'entraînement, le module sauvegarde :

```
polariteModelTensor.keras
tonModelTensor.keras
targetModelTensor.keras
```

Il sauvegarde aussi les objets nécessaires au décodage des labels :

```
polariteLabelTensor.joblib
tonMLBTensor.joblib
targetLabelTensor.joblib
```

Ces fichiers sont indispensables pour faire la prédiction ensuite.

## 14. Statistiques d'entraînement

Le script sauvegarde les résultats d'entraînement dans : `Statistics/tensorTrainingStats.json`

Ce fichier contient :
- les paramètres d'entraînement
- les métriques de polarité
- les métriques de ton
- les métriques de cible

Les métriques principales sont :
- precision
- recall
- f1-score
- support
- accuracy
- macro avg
- weighted avg

Ces statistiques correspondent à l'évaluation interne sur une partie de test issue du corpus annoté automatiquement.

## 15. Fichier `predictionTensor.py`

Le fichier `predictionTensor.py` génère les prédictions finales sur le corpus API simulé.

Il charge : `Data/Processed/frenchTweetAPI.csv`

Puis il charge les modèles entraînés :

```
polariteModelTensor.keras
tonModelTensor.keras
targetModelTensor.keras
```

Il charge aussi :

```
tfidf.joblib
polariteLabelTensor.joblib
tonMLBTensor.joblib
targetLabelTensor.joblib
```

Ensuite, il transforme les textes en vecteurs TF-IDF et applique les 3 modèles.

## 16. Format des prédictions

Les prédictions sont sauvegardées dans : `Data/Processed/predictions.jsonl`

Chaque ligne contient une prédiction au format JSON.

Le fichier contient donc à la fois les labels prédits et une valeur de confiance associée.

## 17. Prédiction multi-label du ton

Le ton utilise un seuil de décision : `MLBTHRESHOLD = 0.5`

Pour chaque ton, si la probabilité est supérieure ou égale au seuil, le label est conservé.

Si aucun ton ne dépasse le seuil, le script retourne : `Autres`
Cela évite d'avoir une prédiction vide.

## 18. Statistiques de prédiction

Après la prédiction, le script affiche et sauvegarde des statistiques dans : `Statistics/predictionTensorStats.json`

Ce fichier contient :
- le nombre total de prédictions
- la distribution des polarités prédites
- la distribution des tons prédits
- la distribution des cibles prédites
- la confiance moyenne pour chaque tâche

Ces statistiques permettent d'observer le comportement global du modèle sur le corpus API.

## 19. Exécution du module

L'exécution peut se faire soit depuis la racine sur chacun des fichiers, soit directement via le pipeline général `python3.12 runSocialAnalyzer.py`

## 20. Résultats obtenus

Les résultats finaux sur le gold standard manuel sont conservés dans le module `FouillesDonnee`.

Les résultats retenus sont environ :

```
Polarité : F1 ≈ 0.665
Ton      : F1 ≈ 0.378
Target   : F1 ≈ 0.865
```

Ces scores montrent que :
- la cible est la tâche la mieux apprise
- la polarité obtient un résultat intermédiaire
- le ton est la tâche la plus difficile

## 21. Interprétation des résultats

### 21.1 Cible

La cible obtient les meilleurs résultats car elle repose souvent sur des indices lexicaux explicites.

Exemples :

```
poutine -> International
macron -> Politique
police -> Sécurité
école -> Education
```

### 21.2 Polarité

La polarité est plus difficile car les opinions sont parfois implicites.

Un tweet peut être purement informatif en apparence mais contenir une critique indirecte ou ironique.

### 21.3 Ton

Le ton est la tâche la plus complexe.

Il dépend fortement :
- du contexte
- de l'ironie
- de la forme du tweet
- des questions rhétoriques
- de la subjectivité de l'annotation

Le ton étant multi-label, son évaluation est également plus stricte.

## 22. Limites du module

Le module possède plusieurs limites :
- les labels d'entraînement viennent d'une weak supervision
- le corpus d'entraînement peut contenir du bruit
- certaines classes sont peu représentées
- l'ironie est très difficile à apprendre
- le TF-IDF ne capture pas toujours le contexte profond
- les modèles MLP restent simples par rapport à des modèles de langue modernes

Ces limites sont normales dans un projet basé sur des tweets courts et des labels faibles.

## 23. Améliorations possibles

Plusieurs améliorations pourraient être envisagées :
- annoter davantage de données manuellement
- utiliser un corpus d'entraînement plus équilibré
- améliorer les règles de weak supervision
- tester des modèles récurrents ou convolutionnels
- utiliser un modèle de langue pré-entraîné adapté au français
- ajuster les seuils multi-label pour le ton
- améliorer la détection de l'ironie.

## 24. Conclusion

Le module `IAApprentissage` constitue le cœur prédictif du projet.

Il transforme les textes en vecteurs TF-IDF, entraîne des modèles Keras/TensorFlow puis génère des prédictions sur un corpus non annoté.

Même si les performances varient selon les tâches, le module permet de mettre en place un pipeline complet d'apprentissage automatique appliqué à l'analyse de tweets politiques.