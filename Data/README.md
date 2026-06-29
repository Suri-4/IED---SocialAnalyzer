# Données – SocialAnalyzer

Ce dossier regroupe les **corpus** utilisés pour SocialAnalyzer.
L'objectif est de conserver une séparation claire entre :
- **corpus d'apprentissage** (utilisé pour la weak supervision et l'entraînement ML)
- **corpus "API simulée"** (utilisé comme jeu de test réaliste, non utilisé lors de l'entraînement)

## Vue d'ensemble des corpus

### Corpus politique (Apprentissage Polarité et Ton)
- **Source brute** : `frenchTweetPolitics.csv` (https://www.kaggle.com/datasets/cibeah/french-tweets-politique)
  Corpus de messages politiques en français.

### Corpus généraliste (Apprentissage TF-IDF)
- **Source brute** : `frenchTweets1-5M.csv` (https://www.kaggle.com/datasets/hbaflast/french-twitter-sentiment-analysis)
  Corpus de tweets français large. Utilisé pour explorer le vocabulaire/TF-IDF.

## Séparation Train / API simulée (70% / 30%)

Afin de simuler un scénario réaliste d'analyse de messages issus d'une API (Twitter/Reddit étant désormais très restrictifs),
le corpus politique est séparé en 2 sous-ensembles :

- **70% -> apprentissage** : utilisé pour produire un corpus annoté via weak supervision puis entraîner le modèle.
- **30% -> API simulée** : conservé à part pour évaluer le modèle sur des données "nouvelles", comme si elles provenaient d'une API.

### Pourquoi garder une approche API Friendly ?
- Le but initial du projet était d'utiliser une API pour fetch les messages d'utilisateur puis ensuite de prédire le ton et la polarité.
- Les API deviennent de plus en plus restrictives (seulement 500 messages par mois via X/Twitter par exemple).
- Garder un fonctionnement pouvant être facilement adapté à une API si cette dernière était mise en place.

### Pourquoi faire cette séparation ?
- éviter tout mélange d'information entre entraînement et test
- simuler un flux réel : nouveaux messages -> prédiction -> résultat (statistiques)

## Normalisation et preprocessing des corpus

Avant toute labellisation ou apprentissage, les corpus sont **normalisés** à l'aide du module `Modules/IngeLangue/normalize.py`.

Cette étape vise à réduire le bruit des données issues de Twitter tout en préservant l'information utile pour l'analyse sociale.

Les principales opérations de normalisation sont :
- mise en minuscules
- suppression des URLs
- remplacement des mentions utilisateurs (`@username`) par le token `@USER`
- suppression des émojis
- réduction de la ponctuation répétée (ex : `!!!` -> `!`)
- normalisation des apostrophes et guillemets (ex : `’` -> `'`)
- retrait des accents unicode
- suppression des espaces multiples.

La normalisation est appliquée :
- au corpus politique **avant** la weak supervision
- au corpus généraliste utilisé pour le **TF-IDF**
- au corpus simulant l'API, afin de garantir une cohérence complète entre apprentissage et prédiction.

## Fichiers produits et rôle de chacun

### Dossier `Processed/`
- `Processed/frenchTweetAPI.csv`  
  Sous-ensemble (30% de `Corpus/frenchTweetPolitics.csv`) réservé à la **simulation d'API**.
  Le corpus est normalisé de la même manière que les données d'apprentissage et ne sera jamais utilisé pendant l'entraînement du modèle.

- `Processed/predictions.jsonl`
  Sortie des prédictions du pipeline d'analyse (résultats ligne à ligne au format JSONL).

### Dossier `Corpus/`
- `LabeledFrenchTweetPolitics.csv`  
  Corpus politique **annoté** par weak supervision (labels polarité + ton + cible) via `Modules/IngeLangue/weakSupervisionLabeler.py`.  
  Sert de base au modèle ML.

- `CorpusFrenchTweets1-5M.csv`
  Corpus généraliste contenant 1.5 million de tweets. Il n'est pas utilisé pour la labellisation, il sert exclusivement à mettre en place une représentation vectorielle **TF-IDF** stable et représentative du français tel qu'il est utilisé sur Twitter.

- `CorpusFrenchTweetPolitics.csv`
  Version formatée (70%) du corpus politique, utilisée comme entrée de la weak supervision.

- `frenchTweetPolitics.csv` et `frenchTweets1-5M.csv`
  Fichiers sources bruts avant formatage.

### Dossier `GoldStandard/`
- `goldStandard.csv`
  Jeu de référence annoté manuellement pour comparaison/évaluation.

- `goldStandardComparison.csv` et `goldStandardComparison.json`
  Résultats de comparaison entre sorties du modèle et gold standard.

## Format standard attendu

Les scripts du projet convergent vers un format commun : `id`, `text`, `polarite`, `ton`, `target`.
- `id` : identifiant unique
- `text` : message original
- `polarite` : `Favorable` / `Défavorable` / `Neutre`
- `ton` : ton mono/multi-label (`Question` / `Argumentatif` / `Ironique` / `Informatif`)
        : Multi-label possible avec séparation `;` (ex : `Question;Informatif`)
- `target` : cible thématique principale (`Politique` / `Sécurité` / `Education` / `International` / `Autre`)

Des statistiques (répartition des labels de polarité, ton et cible) sont calculées après la weak supervision afin de vérifier la cohérence et l'équilibre du corpus annoté avant entraînement.
