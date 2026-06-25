# Module IngeLangue – Ingénierie des Langues

## 1. Présentation du module

Le module `IngeLangue` regroupe les traitements liés à la préparation linguistique des corpus utilisés dans le projet **SocialAnalyzer**.

Son rôle est de transformer des corpus bruts issus de Twitter/X en données textuelles propres, structurées et exploitables par les autres modules du projet, notamment :
- le module de fouille de données
- le module d'apprentissage automatique
- le module de prédiction

Ce module correspond à la partie **Ingénierie des langues** du projet. Il couvre principalement :
- la constitution d'un corpus exploitable
- la normalisation des textes
- le nettoyage du bruit linguistique
- la préparation des données
- l'annotation automatique par supervision faible

## 2. Objectifs

Le module a plusieurs objectifs :

1. Nettoyer les textes issus de corpus sociaux
2. Uniformiser la structure des fichiers CSV
3. Séparer le corpus politique en 2 parties :
   - un corpus d'apprentissage
   - un corpus de prédiction simulant une API
4. Produire un corpus automatiquement annoté
5. Générer des statistiques sur les corpus et les labels

L'objectif général est de préparer une ressource linguistique utilisable par les étapes suivantes du projet.

## 3. Corpus utilisés

Le projet utilise principalement 2 corpus.

### 3.1 Corpus politique

Le fichier `frenchTweetPolitics.csv` contient des tweets politiques en français.

Il est utilisé pour :
- créer un corpus d'entraînement
- créer un corpus de prédiction simulant une API
- produire un corpus annoté automatiquement par weak supervision

Après traitement, ce corpus est séparé en 2 parties :
- `CorpusFrenchTweetPolitics.csv` : partie utilisée pour l'apprentissage
- `frenchTweetAPI.csv` : partie utilisée pour la prédiction finale

### 3.2 Corpus massif de tweets français

Le fichier `frenchTweets1-5M.csv` contient un grand volume de tweets francophones.
Il sert principalement à disposer d'une ressource textuelle plus large pour les traitements statistiques et la représentation du langage (TF-IDF)

## 4. Architecture du module

Le module contient les fichiers principaux suivants :

```
Modules/IngeLangue/
- config.py
- normalize.py
- formatCorpus.py
- weakSupervisionLabeler.py
- README.md
```

## 5. Fichier `config.py`

Le fichier `config.py` centralise les chemins importants du module.

Il définit notamment :
- la racine du projet
- le dossier `Data`
- le dossier des corpus
- le dossier des fichiers traités
- le dossier des statistiques

Cela évite de répéter les chemins dans chaque programme et facilite la maintenance du projet.

Principaux chemins utilisés :

## 6. Fichier `normalize.py`

Le fichier `normalize.py` contient les fonctions de nettoyage et de normalisation des textes.

La fonction principale est : `normalizeText(text)`

Elle applique plusieurs traitements comme :
- passage en minuscules
- normalisation des apostrophes et guillemets
- suppression des URLs
- remplacement des mentions utilisateur
- suppression des accents Unicode
- suppression des emojis
- réduction de la ponctuation répétée
- suppression des espaces multiples

Cette étape est importante car les tweets contiennent souvent beaucoup de bruit : liens, mentions, emojis, caractères spéciaux ou variations typographiques.

## 7. Fichier `formatCorpus.py`

Le fichier `formatCorpus.py` prépare les corpus pour le reste du projet.

Il effectue 2 tâches principales.

### 7.1 Formatage des corpus

La fonction : `formatCorpusCSV(inputPath, outputPath)`

lit un fichier CSV, normalise les textes avec `normalizeText()` puis produit un fichier standardisé avec les colonnes :

```
id,text,polarite,ton,target
```

Ce format facilite l'annotation automatique et l'apprentissage supervisé.

### 7.2 Découpage du corpus politique

La fonction : splitCorpus(inputPath, outPath70, outPath30)

sépare le corpus politique en de2ux parties :

- 70 % pour le corpus d'apprentissage
- 30 % pour le corpus API simulée

Le corpus d'apprentissage conserve les colonnes de labels vides :

```
id,text,polarite,ton,target
```

Le corpus API simulée conserve seulement :

```
id,text
```

Ce découpage permet de simuler une situation réelle : un modèle est entraîné sur une partie du corpus puis utilisé pour prédire les labels de nouveaux textes.

### 7.3 Statistiques de corpus

Le fichier contient aussi la fonction : `saveStatCorpus(file)`

Elle calcule et sauvegarde plusieurs statistiques comme :
- nombre total de lignes
- longueur moyenne des textes
- longueur minimale et maximale
- nombre moyen de mots
- nombre de textes vides
- taille du vocabulaire

Les statistiques sont sauvegardées dans le dossier `Statistics`.

## 8. Fichier `weakSupervisionLabeler.py`

Le fichier `weakSupervisionLabeler.py` applique une annotation automatique par **weak supervision**.

La weak supervision consiste à produire des labels automatiquement à partir de règles simples. Dans ce projet, ces règles reposent principalement sur des listes de mots-clés et un système de score.

Le script annote chaque tweet selon 3 dimensions :
1. la polarité
2. le ton
3. la cible.

## 9. Labels utilisés

### 9.1 Polarité

La polarité correspond à l'orientation générale du message.

Labels utilisés :

```
Favorable
Défavorable
Neutre
```

Le label `Neutre` est utilisé lorsqu'aucun indice clair de polarité favorable ou défavorable n'est détecté.

### 9.2 Ton

Le ton correspond à la manière dont le message est formulé.

Labels utilisés :

```
Question
Argumentatif
Informatif
Ironique
Autres
```

Le ton peut être multi-label. Par exemple, un tweet peut être à la fois :

```
Question;Informatif
```

Le script sélectionne au maximum 2 tons dominants.

### 9.3 Cible

La cible correspond au thème ou à l'objet principal du message.

Labels utilisés :

```
Politique
International
Sécurité
Education
Autre
```

Le label `Autre` est utilisé lorsqu'aucune cible claire n'est détectée.

## 10. Fonctionnement de la weak supervision

Le fonctionnement général est le suivant :
1. Le texte est normalisé avec `normalizeText()`.
2. Le script recherche les mots-clés associés à chaque label.
3. Chaque occurrence trouvée augmente le score du label correspondant.
4. Le label ayant le score le plus élevé est sélectionné.
5. En cas d'égalité, un ordre de priorité est utilisé.
6. Pour le ton, plusieurs labels peuvent être retenus.

## 11. Gestion du multi-label pour le ton

Le ton est traité différemment de la polarité et de la cible.
Pour la polarité et la cible, un seul label est retenu.

Pour le ton, plusieurs labels peuvent être associés à un même texte. Le script applique donc une logique multi-label.
Cela signifie que le script peut retourner jusqu'à 2 tons mais seulement si leurs scores sont suffisamment importants.

## 12. Statistiques de weak supervision

Après l'annotation automatique, le script génère des statistiques sur la distribution des labels.

La fonction : `saveWeakSupervisionStats()`
affiche les statistiques dans le terminal et les sauvegarde dans : Statistics/weakSupervisionStats.json

## 13. Exécution du module

Depuis la racine du projet, les scripts peuvent être lancés individuellement ou directement dans le fonctionnement général comme ceci :

### Annotation automatique par weak supervision

```python3.12 Modules/IngeLangue/weakSupervisionLabeler.py``` ou ```python3.12 runSocialAnalyzer.py```

## 14. Fichiers produits

Le module peut produire plusieurs fichiers.

### Corpus formatés

```
Data/Corpus/CorpusFrenchTweetPolitics.csv
Data/Corpus/CorpusFrenchTweets1-5M.csv
```

### Corpus API simulée

```
Data/Processed/frenchTweetAPI.csv
```

### Corpus annoté automatiquement

```
Data/Corpus/LabeledFrenchTweetPolitics.csv
```

### Statistiques

```
Statistics/frenchTweetPoliticsStats.json
Statistics/CorpusFrenchTweetPoliticsStats.json
Statistics/frenchTweets1-5MStats.json
Statistics/CorpusFrenchTweets1-5MStats.json
Statistics/frenchTweetAPIStats.json
Statistics/weakSupervisionStats.json
```

## 15. Limites du module

La weak supervision repose sur des mots-clés. Elle est donc simple et interprétable mais elle possède plusieurs limites :
- certains mots peuvent être ambigus
- l'ironie est très difficile à détecter automatiquement
- les questions rhétoriques peuvent être confondues avec de vraies questions
- certains tweets sont très courts et manquent de contexte
- la catégorie `Autre` peut absorber des cas difficiles
- les labels produits restent des labels faibles, et non une vérité humaine parfaite.

Ces limites sont importantes : elles expliquent pourquoi une évaluation finale avec un gold standard manuel est nécessaire.

## 16. Justification méthodologique

Ce module permet de montrer plusieurs compétences liées à l'ingénierie des langues :
- constitution d'un corpus
- nettoyage de données textuelles
- normalisation linguistique
- structuration de ressources textuelles
- annotation automatique
- production de statistiques de corpus
- préparation des données pour un système d'apprentissage automatique.

Il constitue donc la base linguistique du projet SocialAnalyzer.

## 17. Améliorations possibles

Plusieurs améliorations pourraient être ajoutées :
- utiliser une vraie tokenisation linguistique
- intégrer une lemmatisation
- mieux gérer les hashtags
- mieux exploiter les emojis au lieu de simplement les supprimer
- ajouter une correction orthographique
- enrichir les règles de weak supervision
- annoter manuellement un plus grand corpus
- utiliser des modèles de langue plus avancés pour mieux détecter l'ironie et les implicites.

## 18. Conclusion

Le module `IngeLangue` assure la préparation linguistique du projet.

Il transforme des tweets bruts en corpus nettoyés, structurés et annotés automatiquement. Ces données servent ensuite aux modules de fouille de données et d'intelligence artificielle.

Même si la weak supervision reste imparfaite, elle permet de produire un corpus annoté exploitable et d'expérimenter un pipeline complet de traitement automatique du langage.