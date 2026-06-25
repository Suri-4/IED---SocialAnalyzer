# Module FouillesDonnee – Fouille de données

## 1. Présentation du module

Le module `FouillesDonnee` regroupe les scripts liés à l'évaluation et à la validation des prédictions produites par le projet **SocialAnalyzer**.

Dans le pipeline général du projet, les textes sont d'abord préparés et annotés automatiquement par le module `IngeLangue` puis les modèles TensorFlow/Keras produisent des prédictions sur un corpus simulant une API. Le module `FouillesDonnee` intervient ensuite pour créer une vérification humaine et comparer les résultats automatiques avec cette référence.

Ce module correspond à la partie **Fouille de données** du projet et met l'accent sur :
- la constitution d'un échantillon d'évaluation 
- la création d'un gold standard manuel 
- la comparaison entre annotations humaines et prédictions automatiques 
- le calcul de métriques de performance 
- l'analyse des limites du modèle

## 2. Objectifs du module

Le module a pour objectif principal de vérifier si les prédictions produites par le modèle sont cohérentes avec une annotation humaine.

Il permet de répondre à plusieurs questions :
- Le modèle retrouve-t-il correctement la polarité d'un tweet ?
- Le modèle identifie-t-il correctement le ton du message ?
- Le modèle détecte-t-il correctement la cible principale du tweet ?
- Les erreurs sont-elles liées à certaines catégories plus difficiles ?
- La weak supervision utilisée pour créer le corpus d'apprentissage est-elle suffisante ?

## 3. Place du module dans le pipeline

Le pipeline général peut être résumé ainsi :

```
Corpus brut
- nettoyage / normalisation
- weak supervision
- apprentissage TensorFlow/Keras
- prédictions sur corpus API
- gold standard manuel
- comparaison et évaluation
```

Le module `FouillesDonnee` intervient surtout dans les 2 dernières étapes.

## 4. Fichiers principaux

Le module contient principalement 2 scripts :

```
Modules/FouillesDonnee/
- goldStandardMaker.py
- goldStandardComparison.py
- README.md
```

## 5. `goldStandardMaker.py`

Le fichier `goldStandardMaker.py` sert à créer un échantillon de tweets à annoter manuellement.

Il lit le corpus API simulé : `Data/Processed/frenchTweetAPI.csv`

Puis il sélectionne aléatoirement 200 tweets avec : `df.sample(n=200, random_state=42)`

Le `random_state=42` permet de rendre l'échantillon reproductible : si le script est relancé sur les mêmes données, il sélectionne les mêmes lignes.

Le fichier généré est : `Data/GoldStandard/goldStandard.csv`

Ce fichier contient les colonnes suivantes :

```
id
text
polariteGold
tonGold
targetGold
commentaireGold
```

Les colonnes `polariteGold`, `tonGold`, `targetGold` et `commentaireGold` sont laissées vides au départ. Elles doivent ensuite être remplies manuellement.

## 6. Rôle du gold standard

Il sert à comparer les prédictions du modèle avec des labels humains. C'est une étape importante car les labels utilisés pour l'entraînement sont issus de weak supervision donc ils ne sont pas parfaits.

Le gold standard permet donc de mesurer plus honnêtement les performances du modèle.

Dans ce projet, le gold standard contient 200 tweets issus du corpus API simulé. Ce choix permet d'évaluer le modèle sur des tweets qui n'ont pas servi directement à l'entraînement.

## 7. Labels utilisés dans le gold standard

### 7.1 Polarité

La polarité correspond à l'orientation générale du message.

Labels utilisés :

```
Favorable
Neutre
Défavorable
```

- `Favorable` : le tweet exprime un soutien ou une opinion positive.
- `Défavorable` : le tweet exprime une critique ou une opinion négative.
- `Neutre` : le tweet est surtout informatif ou ne contient pas d'opinion claire.

### 7.2 Ton

Le ton correspond à la manière dont le message est formulé.

Labels utilisés :

```
Informatif
Argumentatif
Question
Ironique
Autres
```

Le ton peut être multi-label. Par exemple :

```
Question;Argumentatif
```

### 7.3 Cible

La cible correspond au thème principal du tweet.

Labels utilisés :

```
Politique
International
Sécurité
Education
Autre
```

- `Politique` : institutions, partis, élus, gouvernement, élections françaises, réformes.
- `International` : pays étrangers, relations internationales, Russie, Ukraine, Poutine, Europe, etc.
- `Sécurité` : police, justice, armée, violence, sécurité publique.
- `Education` : école, université, enseignants, élèves, formation.
- `Autre` : aucun thème principal clair parmi les catégories précédentes.

## 8. `goldStandardComparison.py`

Le fichier `goldStandardComparison.py` compare le gold standard manuel avec les prédictions produites par les modèles.

Il utilise 2 fichiers :

```
Data/GoldStandard/goldStandard.csv
Data/Processed/predictions.jsonl
```

La comparaison se fait grâce à la colonne `id`.

Le script fusionne donc les 2 fichiers pour comparer les mêmes tweets possèdant le même ID

Ensuite, il compare :

```
polariteGold <-> polaritePred
tonGold <-> tonPred
targetGold <-> targetPred
```

## 9. Métriques calculées

Le script calcule plusieurs métriques comme :
-Précision
-Rappel
-F1
-Exact match
-Partial match

### 9.1 Précision

La précision mesure la part des prédictions positives qui sont correctes.

### 9.2 Rappel

Le rappel mesure la capacité du modèle à retrouver les bons labels attendus.

### 9.3 F1-score

Le F1-score est une moyenne entre précision et rappel. C'est une métrique utile lorsque les classes sont déséquilibrées.

### 9.4 Exact match

L'exact match vérifie si la prédiction correspond exactement au gold standard.

Principalement utilisé pour le ton.

Exemple :

```
Gold : Question;Argumentatif
Pred : Question
exact match : faux
```

Même si le modèle a retrouvé une partie du bon ton.

### 9.5 Partial match

Le partial match est surtout utile pour le ton multi-label.

Il considère qu'une prédiction est partiellement correcte si au moins un label est commun entre le gold standard et la prédiction.

Exemple :

```
Gold : QuestionArgumentatif
Pred : Question
partial match : vrai
```

Cette métrique permet de ne pas pénaliser totalement le modèle lorsqu'il retrouve une partie du ton attendu.

## 10. Sauvegarde des statistiques

Après l'évaluation, le script sauvegarde les résultats dans : `Statistics/goldStandardEvaluationStats.json`

Ce fichier contient :
- le nombre de lignes dans le gold standard 
- le nombre de prédictions 
- le nombre de tweets comparés 
- les scores pour la polarité 
- les scores pour le ton 
- les scores pour la cible

## 11. Exécution du module

### Créer le gold standard

Depuis la racine du projet : `python3.12 Modules/FouillesDonnee/goldStandardMaker.py`

Cela génère : `Data/GoldStandard/goldStandard.csv`

Attention : si le fichier a déjà été annoté manuellement, il ne faut pas relancer ce script sans sauvegarde car il pourrait écraser le travail d'annotation.

### Comparer le gold standard avec les prédictions

Après avoir généré les prédictions avec le module IA, nous pouvons comparer les résultats avec : `python3.12 Modules/FouillesDonnee/goldStandardComparison.py`

Cela affiche les résultats dans le terminal et sauvegarde les statistiques dans : `Statistics/goldStandardEvaluationStats.json`

## 12. Résultats finaux obtenus

L'évaluation finale sur le gold standard montre des performances différentes selon les tâches.

Les résultats retenus sont :

```
Polarité : F1 ≈ 0.665
Ton      : F1 ≈ 0.378
Target   : F1 ≈ 0.865
```

Ces résultats montrent que la cible thématique est la tâche la mieux apprise par le modèle.

La polarité obtient un résultat intermédiaire car certains tweets sont ambigus : un tweet peut sembler neutre tout en contenant une critique implicite.

Le ton est la tâche la plus difficile car il est parfois multi-label et dépend fortement du contexte. L'ironie, les questions rhétoriques et les formulations courtes sont particulièrement difficiles à détecter.

## 13. Analyse des résultats

### 13.1 Cible

La cible obtient le meilleur score.

Cela s'explique par le fait que les catégories thématiques reposent souvent sur des indices lexicaux explicites :
- noms de pays 
- personnalités politiques 
- institutions 
- termes liés à la sécurité 
- termes liés à l'éducation

### 13.2 Polarité

La polarité est plus difficile.

Certaines opinions sont explicites mais d'autres sont implicites ou dépendent du contexte.

Par exemple, un tweet peut relayer une information tout en sous-entendant une critique.

### 13.3 Ton

Le ton est la tâche la plus complexe du projet.

Plusieurs raisons expliquent cela :
- le ton peut être multi-label 
- l'ironie dépend du contexte 
- les tweets sont courts 
- les questions peuvent être rhétoriques 
- `Autres` peut regrouper des cas très différents

Le score plus faible sur le ton ne signifie donc pas que le pipeline ne fonctionne pas. Il montre plutôt que cette tâche est plus subjective et plus difficile à automatiser.

## 14. Limites du module

Le module possède plusieurs limites :
- le gold standard ne contient que 200 tweets 
- l'annotation humaine peut elle-même contenir des hésitations 
- certaines catégories sont peu représentées 
- le ton est difficile à évaluer avec une correspondance exacte 
- la comparaison dépend de la qualité des prédictions générées en amont 
- les labels d'entraînement restent issus d'une weak supervision

## 15. Améliorations possibles

Plusieurs améliorations sont possibles :
- annoter un gold standard plus grand 
- faire relire le gold standard par plusieurs annotateurs 
- enrichir l'analyse des erreurs ligne par ligne 
- améliorer la détection du ton 
- ajuster les seuils de prédiction multi-label 
- utiliser des représentations textuelles plus avancées

## 16. Conclusion

Le module `FouillesDonnee` permet de valider expérimentalement le pipeline du projet.

Il ne se contente pas de produire des prédictions, il les compare à une annotation humaine, calcule des métriques et met en évidence les limites du système.

Ce module montre donc une démarche complète de fouille de données :

-constitution d'un échantillon
- annotation manuelle
- comparaison
- métriques
- analyse des résultats

Il constitue une étape essentielle pour évaluer la qualité réelle du modèle.