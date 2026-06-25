# SocialAnalyzer - Analyse de sentiments multi-labels

SocialAnalyzer est un projet Python de l'IED Paris 8 pour analyser automatiquement des tweets en francais

Le pipeline couvre toute la chaine :
- préparation des corpus
- annotation automatique (weak supervision)
- apprentissage TF-IDF + reseaux de neurones Keras
- prédiction sur un corpus "API simulée"
- évaluation sur gold standard manuel
- génération de statistiques JSON

## Objectif

Predire, pour chaque message :
- une polarite : `Favorable`, `Defavorable`, `Neutre`
- un ton (multi-label) : `Argumentatif`, `Question`, `Informatif`, `Ironique`, `Autres`
- une cible thematique : `Politique`, `Securite`, `Education`, `International`, `Autre`

## Pipeline du projet

1. **Formatage des corpus**
   - Script : `Modules/IngeLangue/formatCorpus.py`
   - Normalise les textes (URLs, mentions, emojis, accents, ponctuation répétitive, etc.)
   - Produit :
     - `Data/Corpus/CorpusFrenchTweetPolitics.csv`
     - `Data/Corpus/CorpusFrenchTweets1-5M.csv`
     - `Data/Processed/frenchTweetAPI.csv` (split 30% API simulée)
   - Génère des statistiques de corpus dans `Statistics/`

2. **Weak supervision (annotation automatique)**
   - Script : `Modules/IngeLangue/weakSupervisionLabeler.py`
   - Ajoute `polarite`, `ton`, `target` via dictionnaires de mots-clés
   - Produit : `Data/Corpus/LabeledFrenchTweetPolitics.csv`
   - Génère : `Statistics/weakSupervisionStats.json`

3. **Apprentissage du TF-IDF**
   - Script : `Modules/IAApprentissage/ML/trainTFIDF.py`
   - Corpus : `Data/Corpus/CorpusFrenchTweets1-5M.csv`
   - Produit : `Modules/IAApprentissage/ML/TrainedModels/tfidf.joblib`

4. **Entrainement des modeles TensorFlow/Keras**
   - Script : `Modules/IAApprentissage/ML/trainTensorLabel.py`
   - Corpus : `Data/Corpus/LabeledFrenchTweetPolitics.csv`
   - Entraine 3 modeles : polarite, ton (multi-label), target
   - Produits :
     - `Modules/IAApprentissage/ML/TrainedModels/polariteModelTensor.keras`
     - `Modules/IAApprentissage/ML/TrainedModels/tonModelTensor.keras`
     - `Modules/IAApprentissage/ML/TrainedModels/targetModelTensor.keras`
     - encodeurs labels (`*.joblib`)
   - Génère : `Statistics/tensorTrainingStats.json`

5. **Prediction sur corpus API simulee**
   - Script : `Modules/IAApprentissage/ML/predictionTensor.py`
   - Entrée : `Data/Processed/frenchTweetAPI.csv`
   - Sortie : `Data/Processed/predictions.jsonl`
   - Génère : `Statistics/predictionTensorStats.json`

6. **Evaluation sur gold standard manuel**
   - Echantillonnage : `Modules/FouillesDonnee/goldStandardMaker.py`
   - Comparaison : `Modules/FouillesDonnee/goldStandardComparison.py`
   - Fichiers :
     - `Data/GoldStandard/goldStandard.csv`
     - `Data/GoldStandard/goldStandardComparison.csv`
     - `Data/GoldStandard/goldStandardComparison.json`
   - Génère : `Statistics/goldStandardEvaluationStats.json`

## Execution rapide

Depuis la racine du projet :

`python runSocialAnalyzer.py`

Le script principal enchaine :
- formatage
- weak supervision
- entrainement TF-IDF
- entrainement des modeles labels
- prédiction API

Options disponibles :

`python runSocialAnalyzer.py --skipFormatting --skipLabeling --skipTFIDF --skipLabelTraining --skipPrediction`

## Evaluation manuelle (gold standard)

1. Créer l'echantillon à annoter :

`python Modules/FouillesDonnee/goldStandardMaker.py`

2. Renseigner manuellement les colonnes `polariteGold`, `tonGold`, `targetGold` dans `Data/GoldStandard/goldStandard.csv`

3. Comparer avec les predictions :

`python Modules/FouillesDonnee/goldStandardComparison.py`

## Technologies utilisees (reelles)

- Python 3
- pandas, numpy
- scikit-learn (TF-IDF, metriques, pre-processing labels)
- TensorFlow/Keras (modeles de classification)
- joblib
- emoji

## Structure actuelle

SocialAnalyzer/
├── Data/
│   ├── Corpus/
│   ├── GoldStandard/
│   └── Processed/
├── Modules/
│   ├── FouillesDonnee/
│   ├── IAApprentissage/ML/
│   └── IngeLangue/
├── Statistics/
├── runSocialAnalyzer.py
└── README.md


## Remarques

- Le ton est traite en **multi-label** (plusieurs tons possibles pour un même message).
- La source d'annotation d'entrainement repose sur la **weak supervision** (regles lexicales), puis est evaluée via un gold standard manuel.
- Les statistiques sont exportées en JSON dans `Statistics/` pour exploitation.

