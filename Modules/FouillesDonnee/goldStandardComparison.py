# --------------------------------------------------------------
# Nom .........: goldStandardComparison.py
# Auteur ......: Sacha Deliège
# Date ........: Mai 2026
# But .........: Comparer le gold standard manuel avec les prédictions
# .............: du modèle TensorFlow/Keras.
# Compilation .: python Modules/FouillesDonnee/goldStandardComparison.py
# --------------------------------------------------------------

import json
import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# Chemins des fichiers
GOLDSTANDARDPATH = Path("Data/GoldStandard/goldStandard.csv")
PREDICTIONSPATH = Path("Data/Processed/predictions.jsonl")
STATSOUTPUTPATH = Path("Statistics/goldStandardEvaluationStats.json")

# MARK: parseLabels()
# Transforme une valeur en ensemble de labels (pour le MultiLabelBinarizer)
# Exemple : "Question;Informatif" devient {"Question", "Informatif"}
def parseLabels(value):
	if isinstance(value, (list, tuple, set)):
		values = list(value)
	elif hasattr(value, "tolist") and not isinstance(value, str):
		values = value.tolist()
		if not isinstance(values, list):
			values = [values]
	else:
		if pd.isna(value):
			return set()

		values = str(value).split(";")

	labels = set()

	for label in values:
		label = str(label).strip()
		if label:
			labels.add(label)

	return labels

# MARK: partialMatch()
# Vérifie si au moins un label est commun entre le gold et la prédiction
def partialMatch(goldLabels, predLabels):
	# Si l'un des deux ensembles est vide, aucun match n'est possible.
	if not goldLabels or not predLabels:
		return False

	# On cherche un label du gold présent aussi dans la prédiction.
	for label in goldLabels:
		if label in predLabels:
			return True

	# Aucun label commun trouvé.
	return False

# MARK: evaluateCategory()
# Calcule les scores principaux pour une catégorie
def evaluateCategory(comparison, category):
	# Récupération des labels du gold standard et des prédictions
	goldColumn = f"{category}Gold" 
	predColumn = f"{category}Pred"

	# Transformation des valeurs en ensembles de labels
	goldLabels = comparison[goldColumn].apply(parseLabels)
	predLabels = comparison[predColumn].apply(parseLabels)

	# Récupération de tous les labels possibles
	allLabels = sorted(set().union(*goldLabels, *predLabels))

	# Si aucun label n'est présent dans le gold ou les prédictions, on retourne des scores à 0
	if not allLabels: 
		return {
			"precision": 0,
			"recall": 0,
			"f1": 0,
			"exactMatch": 0,
			"partialMatch": 0
		}

	# Transformation des ensembles de labels en tableaux binaires
	mlb = MultiLabelBinarizer(classes=allLabels)
	yTrue = mlb.fit_transform(goldLabels)
	yPred = mlb.transform(predLabels)

	# Scores classiques
	precision = precision_score(yTrue, yPred, average="micro", zero_division=0)
	recall = recall_score(yTrue, yPred, average="micro", zero_division=0)
	f1 = f1_score(yTrue, yPred, average="micro", zero_division=0)
	exactMatch = accuracy_score(yTrue, yPred)

	# Score plus souple : au moins un label correct
	partialResults = []

	# On compare chaque ligne du gold et de la prédiction pour voir s'il y a au moins un label en commun
	for gold, pred in zip(goldLabels, predLabels):
		partialResults.append(partialMatch(gold, pred))

	partialMatchScore = sum(partialResults) / len(partialResults)

	return {
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
		"exactMatch": float(exactMatch),
		"partialMatch": float(partialMatchScore)
	}

# MARK: printCategoryResult()
# Affiche les résultats d'une catégorie
def printCategoryResult(category, result):
	print(f"\nCatégorie : {category}")
	print(f"Précision ....: {float(result['precision'])}")
	print(f"Rappel .......: {float(result['recall'])}")
	print(f"F1 ...........: {float(result['f1'])}")
	print(f"Exact match ..: {float(result['exactMatch'])}")

	if category == "ton":
		print(f"Partial match : {float(result['partialMatch'])}")

# MARK: saveStats()
# Sauvegarde les résultats dans un fichier JSON
def saveStats(results, totalGold, totalPredictions, totalCompared):
	report = {
		"file": "goldStandardComparison.py",
		"generatedAt": datetime.datetime.now().isoformat(),
		"totalGoldRows": totalGold,
		"totalPredictionRows": totalPredictions,
		"totalComparedRows": totalCompared,
		"results": results
	}

	STATSOUTPUTPATH.parent.mkdir(parents=True, exist_ok=True)

	with open(STATSOUTPUTPATH, "w", encoding="utf-8") as fOUT:
		json.dump(report, fOUT, ensure_ascii=False, indent=4)

	print(f"\nStatistiques sauvegardées dans : {STATSOUTPUTPATH}")

# MARK: main()
def main():
	# Vérification de l'existence des fichiers
	if not GOLDSTANDARDPATH.exists():
		raise FileNotFoundError(f"Fichier introuvable : {GOLDSTANDARDPATH}")
	if not PREDICTIONSPATH.exists():
		raise FileNotFoundError(f"Fichier introuvable : {PREDICTIONSPATH}")

	# Lecture des fichiers
	goldStandard = pd.read_csv(GOLDSTANDARDPATH)
	predictions = pd.read_json(PREDICTIONSPATH, lines=True)

	# Colonnes obligatoires
	requiredGoldColumns = ["id", "polariteGold", "tonGold", "targetGold"]
	requiredPredictionColumns = ["id", "polaritePred", "tonPred", "targetPred"]

	# Vérification de la présence des colonnes nécessaires
	for column in requiredGoldColumns:
		if column not in goldStandard.columns:
			raise ValueError(f"Colonne absente dans le gold standard : {column}")
	for column in requiredPredictionColumns:
		if column not in predictions.columns:
			raise ValueError(f"Colonne absente dans les prédictions : {column}")

	# Fusion des deux fichiers grâce à l'id
	comparison = pd.merge(goldStandard, predictions, on="id")

	print("===== ÉVALUATION GOLD STANDARD =====")
	print(f"Tweets dans le gold standard : {len(goldStandard)}")
	print(f"Tweets dans les prédictions .: {len(predictions)}")
	print(f"Tweets comparés .............: {len(comparison)}")

	# Évaluation des trois catégories
	results = {}

	for category in ["polarite", "ton", "target"]:
		results[category] = evaluateCategory(comparison, category)
		printCategoryResult(category, results[category])

	# Sauvegarde JSON
	saveStats(
		results,
		len(goldStandard),
		len(predictions),
		len(comparison)
	)

if __name__ == "__main__":
	main()