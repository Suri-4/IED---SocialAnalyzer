# --------------------------------------------------------------
# Nom ...: predictionTensor.py
# Auteur : Sacha Deliège
# Date ..: Avril 2026
# But ...: Générer des prédictions polarité + ton + cible
# .......: sur un corpus non annoté (API simulée après formatage)
# .......: en utilisant les modèles TensorFlow/Keras déjà entraînés
# --------------------------------------------------------------

import json
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
import datetime

from config import (
	TFIDFMODELPATH,
	APICORPUSPATH,
	PREDICTIONOUTPUTPATH,
	SAVEDFOLDER,
	STATISTICFOLDER
)

MLBTHRESHOLD = 0.5 # Seuil de confiance pour les prédictions multi-label (ton)

# MARK: percentage()
# Calcule un pourcentage avec gestion du cas total = 0.
def percentage(count, total):
    if total <= 0:
        return 0

    valuePercentage = (count/total)*100 # Calcul du pourcentage
    return valuePercentage

# MARK: buildDistributionStats()
# Fonction pour transformer une distribution pandas en dictionnaire JSON
def buildDistributionStats(counts):
	total = int(counts.sum())
	distribution = {} # Dictionnaire pour stocker les stats de chaque label

	# Calcul du pourcentage pour chaque label et stockage dans le dictionnaire
	for label, count in counts.items():
		labelPercentage = 0
		if total > 0:
			labelPercentage = percentage(count, total)

		distribution[str(label)] = {
			"count": int(count),
			"percentage": labelPercentage
		}

	return {
		"total": total,
		"distribution": distribution
	}


# MARK: savePredictionStats()
# Fonction pour sauvegarder les statistiques de prédiction TensorFlow/Keras
def savePredictionStats(predictions, polariteCounts, tonCounts, targetCounts):
	polariteAccuracy = [predi["polariteAcc"] for predi in predictions]
	tonAccuracy = [predi["tonAcc"] for predi in predictions]
	targetAccuracy = [predi["targetAcc"] for predi in predictions]

	# Construction du rapport de statistiques
	report = {
		"file": PREDICTIONOUTPUTPATH.name,
		"generatedAt": datetime.datetime.now().isoformat(),
		"totalPredictions": len(predictions),

		"polarite": buildDistributionStats(polariteCounts),
		"ton": buildDistributionStats(tonCounts),
		"target": buildDistributionStats(targetCounts),

		"confidence": {
			"polariteMean": float(np.mean(polariteAccuracy)) if polariteAccuracy else 0,
			"tonMean": float(np.mean(tonAccuracy)) if tonAccuracy else 0,
			"targetMean": float(np.mean(targetAccuracy)) if targetAccuracy else 0
		}
	}

	STATISTICFOLDER.mkdir(parents=True, exist_ok=True)
	reportPath = STATISTICFOLDER / "predictionTensorStats.json"

	with open(reportPath, "w", encoding="utf-8") as fOUT:
		json.dump(report, fOUT, ensure_ascii=False, indent=4)

	print(f"Statistiques de prédiction sauvegardées dans : {reportPath}")

# MARK: npToPy()
# Fonction pour convertir les types numpy en types natifs Python lors de l'écriture JSON
def npToPy(obj):
	if isinstance(obj, np.integer): # Convertit les types numpy en types Python natifs
		return int(obj)
	if isinstance(obj, np.floating):
		return float(obj)
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	return obj


# MARK: savePrediction()
# Fonction pour sauvegarder les prédictions dans un fichier JSONL
def savePrediction(predictions, outPath):
	outPath.parent.mkdir(parents=True, exist_ok=True)
	with outPath.open("w", encoding='utf-8') as fOUT:
		for pred in predictions:
			json.dump(pred, fOUT, default=npToPy, ensure_ascii=False)
			fOUT.write("\n")

# MARK: decodeMultilabelPrediction()
# Fonction pour décoder les prédictions multi-label en labels 
def decodeMultilabelPrediction(probabilities, classes, MLBTHRESHOLD):
	# Sélection des indices des labels dont la probabilité dépasse le seuil
	selectedIndices = np.where(probabilities >= MLBTHRESHOLD)[0]

	# Récupération des labels correspondants à ces indices
	selectedLabels = []
	for index in selectedIndices:
		selectedLabels.append(classes[index])

	# Si aucun label n'est sélectionné, on retourne "Autres" avec la meilleure confiance globale
	if not selectedLabels:
		bestOverallConfidence = float(np.max(probabilities))
		return ["Autres"], bestOverallConfidence

	# Sinon, on retourne les labels retenus et la meilleure confiance parmi eux
	bestSelectedConfidence = float(np.max(probabilities[selectedIndices]))
	return selectedLabels, bestSelectedConfidence

# MARK: main()
def main():
	# Chargement des modèles entraînés
	vectorizer = joblib.load(TFIDFMODELPATH)
	polariteModel = load_model(SAVEDFOLDER / "polariteModelTensor.keras")
	tonModel = load_model(SAVEDFOLDER / "tonModelTensor.keras")
	targetModel = load_model(SAVEDFOLDER / "targetModelTensor.keras")

	tonMLB = joblib.load(SAVEDFOLDER / "tonMLBTensor.joblib")
	polariteEncoder = joblib.load(SAVEDFOLDER / "polariteLabelTensor.joblib")
	targetEncoder = joblib.load(SAVEDFOLDER / "targetLabelTensor.joblib")

	# Charger le corpus API simulé
	apiCorpus = pd.read_csv(APICORPUSPATH)
	if 'text' not in apiCorpus.columns:
		raise ValueError(f"Colonne 'text' absente dans {APICORPUSPATH}. Colonnes disponibles : {list(apiCorpus.columns)}")

	# Représentation TF-IDF puis conversion dense pour Keras
	tfidfMatrix = vectorizer.transform(apiCorpus['text'].fillna("").astype(str))
	features = tfidfMatrix.toarray()

	# Prédictions des 3 modèles
	polariteProb = polariteModel.predict(features) # Probabilités de polarité pour chaque classe 
	targetProb = targetModel.predict(features) # Probabilités de cible pour chaque classe
	tonProb = tonModel.predict(features) # Probabilités de ton pour chaque classe (multi-label)

	# Décodage des prédictions en labels
	polariteIdx = np.argmax(polariteProb, axis=1) # Indices des classes de polarité avec la plus haute probabilité 
	targetIdx = np.argmax(targetProb, axis=1) # Indices des classes de cible avec la plus haute probabilité
	polaritePred = polariteEncoder.inverse_transform(polariteIdx) # Décodage des idx de polarité en labels
	targetPred = targetEncoder.inverse_transform(targetIdx)

	# Préparer les prédictions pour l'écriture
	predictions = []
	for i in range(len(apiCorpus)):
		# Décodage multi-label pour le ton
		tonPredLabels, tonConfidence = decodeMultilabelPrediction(tonProb[i], tonMLB.classes_, MLBTHRESHOLD)

		# Construction de la prédiction
		pred = {
			"id": apiCorpus.loc[i, 'id'],
			"text": apiCorpus.loc[i, 'text'],
			"polaritePred": polaritePred[i],
			"polariteAcc": float(np.max(polariteProb[i])),
			"tonPred": tonPredLabels,
			"tonAcc": tonConfidence,
			"targetPred": targetPred[i],
			"targetAcc": float(np.max(targetProb[i]))
		}
		predictions.append(pred)

	# Sauvegarder les prédictions
	savePrediction(predictions, PREDICTIONOUTPUTPATH)
	print(f"Predictions sauvegardees dans : {PREDICTIONOUTPUTPATH}")

	# Statistiques
	print("\n===== STATISTIQUES DES PREDICTIONS (TENSOR) =====")
	print(f"Nombre de predictions : {len(predictions)}")

	polariteCounts = pd.Series([predi['polaritePred'] for predi in predictions]).value_counts()
	print("\nDistribution des classes de polarite :")
	print(polariteCounts)

	tonFlat = [label for predi in predictions for label in predi['tonPred']]
	tonCounts = pd.Series(tonFlat).value_counts()
	print("\nDistribution des classes de ton :")
	print(tonCounts)

	targetCounts = pd.Series([predi['targetPred'] for predi in predictions]).value_counts()
	print("\nDistribution des classes de cible :")
	print(targetCounts)

	polariteAccs = [predi['polariteAcc'] for predi in predictions]
	tonAccs = [predi['tonAcc'] for predi in predictions]
	targetAccs = [predi['targetAcc'] for predi in predictions]
	print(f"\nConfiance moyenne polarite : {float(np.mean(polariteAccs))}")
	print(f"Confiance moyenne ton : {float(np.mean(tonAccs))}")
	print(f"Confiance moyenne cible : {float(np.mean(targetAccs))}")

	savePredictionStats(predictions, polariteCounts, tonCounts, targetCounts)

if __name__ == "__main__":
	main()

