# --------------------------------------------------------------
# Nom ...: trainTensorLabel.py
# Auteur : Sacha Deliège
# Date ..: Avril 2026
# But ...: Entraîner trois modèles supervisés (polarité, ton, cible)
# .......: en utilisant le TF-IDF préalablement appris
# .......: et des réseaux de neurones Dense (MLP) via TensorFlow/Keras
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import joblib
import json
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping

from config import (
	TFIDFMODELPATH,
	LABELEDCORPUSPATH,
	SAVEDFOLDER,
	STATISTICFOLDER,
)

# Constantes d'entraînement
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
EPOCHS = 40
BATCH_SIZE = 64
TON_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 6

# MARK: splitLabel()
# Fonction pour séparer les labels multi-labels (ex: "ironique;humoristique") en liste de labels individuels
def splitLabel(labelStr):
	# Gérer les valeurs manquantes ou NaN
	if pd.isna(labelStr): 
		return []

	# Séparer les labels par ';' et nettoyer les espaces
	labels = str(labelStr).split(';')
	labelsParsed = []
	for label in labels:
		label = label.strip()
		if label:
			labelsParsed.append(label) # Ajouter le label à la liste des labels parsés
	return labelsParsed 

# MARK: buildEarlyStopping()
# Fonction pour créer un EarlyStopping pour chaque modèle
def buildEarlyStopping():
	return EarlyStopping(
		monitor='val_loss', # Surveiller la perte de validation pour détecter le surapprentissage
		patience=EARLY_STOPPING_PATIENCE, # Nombre d'époques sans amélioration avant d'arrêter l'entraînement
		restore_best_weights=True, # Restaurer les poids du meilleur modèle trouvé pendant l'entraînement
	)

# MARK: buildMulticlassModel()
# Construit un modèle Keras pour une classification multi-classe
def buildMulticlassModel(inputDim, outputDim):
	# Modèle à 2 couches cachées Relu avec Dropout pour la régularisation, et une couche de sortie softmax pour la classification multi-classe
	model = Sequential([
		Input(shape=(inputDim,)),
		Dense(512, activation='relu'), # Relu pour éviter une linéarité trop forte
		Dropout(0.2), # Réduction du surapprentissage en désactivant aléatoirement 20% des neurones pendant l'entraînement
		Dense(256, activation='relu'),
		Dropout(0.2),
		Dense(outputDim, activation='softmax') # Softmax pour obtenir des probabilités de classe
	])

	model.compile(
		optimizer='adam', # Adam est un optimiseur efficace pour la plupart des tâches de classification
		loss='sparse_categorical_crossentropy', # Perte adaptée pour la classification multi-classe avec des labels encodés en entiers
		metrics=['accuracy'] # Mesure de performance principale : l'exactitude (accuracy)
	)
	return model


# MARK: buildMultilabelModel()
# Construit un modèle Keras pour une classification multi-label
def buildMultilabelModel(inputDim, outputDim):
	model = Sequential([
		Input(shape=(inputDim,)),
		Dense(512, activation='relu'),
		Dropout(0.2),
		Dense(256, activation='relu'),
		Dropout(0.2),
		Dense(outputDim, activation='sigmoid') # Sigmoid pour obtenir des probabilités indépendantes pour chaque classe dans la classification multi-label
	])

	model.compile(
		optimizer='adam',
		loss='binary_crossentropy', # Perte binaire adaptée pour la classification multi-label où chaque classe est traitée indépendamment
		metrics=['binary_accuracy'] # binary_accuracy évalue la précision pour chaque classe indépendamment
	)
	return model

# MARK: saveTrainingStats()
# Fonction pour sauvegarder les statistiques d'entraînement TensorFlow/Keras
def saveTrainingStats(polariteReport, tonReport, targetReport):
	# Construction du rapport d'entraînement
	report = {
		"file": "trainTensorLabel.py",
		"generatedAt": datetime.datetime.now().isoformat(),
		"parameters": {
			"randomState": RANDOM_STATE, # Valeur de reproductibilité pour les splits de données
			"testSize": TEST_SIZE, # Proportion de données réservée pour le test
			"validationSplit": VALIDATION_SPLIT, # Proportion de données d'entraînement réservée pour la validation pendant l'entraînement
			"epochs": EPOCHS, # Nombre maximum d'époques pour l'entraînement
			"batchSize": BATCH_SIZE, # Nombre d'échantillons par lot pour l'entraînement
			"tonThreshold": TON_THRESHOLD, # Seuil de confiance pour les prédictions multi-label du ton
			"earlyStoppingPatience": EARLY_STOPPING_PATIENCE, # Nombre d'époques sans amélioration avant d'arrêter l'entraînement
		},

		"polarite": polariteReport, # Rapport de classification pour la polarité
		"ton": tonReport, # Rapport de classification pour le ton
		"target": targetReport, # Rapport de classification pour la cible
	}

	STATISTICFOLDER.mkdir(parents=True, exist_ok=True)
	reportPath = STATISTICFOLDER / "tensorTrainingStats.json"

	# Sauvegarde du rapport d'entraînement au format JSON
	with open(reportPath, "w", encoding="utf-8") as fOUT:
		json.dump(report, fOUT, ensure_ascii=False, indent=4)

	print(f"Statistiques d'entraînement sauvegardées dans : {reportPath}")

# MARK: main()
def main():
	# Chargement du corpus labellisé
	corpusData = pd.read_csv(LABELEDCORPUSPATH)
	requiredCols = {'text', 'polarite', 'ton', 'target'}
	if not requiredCols.issubset(corpusData.columns):
		raise ValueError(f"Colonnes attendues manquantes dans {LABELEDCORPUSPATH} : {list(corpusData.columns)}")

	# Chargement du vectorizer TF-IDF déjà appris
	vectorizer = joblib.load(TFIDFMODELPATH)
	tfidfMatrix = vectorizer.transform(corpusData['text'].fillna('').astype(str))
	features = tfidfMatrix.toarray()

	# Préparation des labels polarité et target
	polariteEncoder = LabelEncoder() # Encodage des labels de polarité en entiers
	targetEncoder = LabelEncoder() # Encodage des labels de cible en entiers

	# Encodage des labels de polarité et cible, en gérant les valeurs manquantes
	labelPolarite = polariteEncoder.fit_transform(corpusData['polarite'].fillna('Neutre').astype(str))
	labelTarget = targetEncoder.fit_transform(corpusData['target'].fillna('Autre').astype(str))

	# Préparation du ton en multi-label
	ton = corpusData['ton'].fillna('').apply(splitLabel) # Séparation des labels multi-labels en listes de labels individuels
	tonMLB = MultiLabelBinarizer() # Binarisation des labels de ton pour la classification multi-label (ex: "Informatif;Question" devient [1, 0, 1, 0, ...] selon les classes)
	labelTon = tonMLB.fit_transform(ton)

	# Split train/test commun aux trois tâches.
	(featureTrain, featureTest, polariteTrain, polariteTest, tonTrain, tonTest, targetTrain, targetTest) = train_test_split(
		features, labelPolarite, labelTon, labelTarget, test_size=TEST_SIZE, stratify=labelPolarite, random_state=RANDOM_STATE)

	# MARK: POLARITE
	print("\n===== ENTRAÎNEMENT POLARITÉ =====")

	# Construction du modèle de polarité
	polariteModel = buildMulticlassModel( 
		featureTrain.shape[1], # Nombre de caractéristiques d'entrée (dimensions du TF-IDF)
		len(polariteEncoder.classes_) # Nombre de classes de polarité à prédire (ex: Positif, Négatif, Neutre)
	)

	# Entraînement du modèle de polarité
	polariteModel.fit(
		featureTrain, # Caractéristiques d'entraînement (TF-IDF)
		polariteTrain, # Labels d'entraînement pour la polarité (entiers encodés)
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_split=VALIDATION_SPLIT,
		callbacks=[buildEarlyStopping()]
	)

	# Prédiction et évaluation du modèle de polarité
	polariteProb = polariteModel.predict(featureTest)
	polaritePredict = polariteEncoder.inverse_transform(np.argmax(polariteProb, axis=1))
	polariteTruth = polariteEncoder.inverse_transform(polariteTest)

	print("\n--- Rapport Polarite (TensorFlow) ---")
	print(classification_report(polariteTruth, polaritePredict, zero_division=0))
	polariteReport = classification_report(polariteTruth, polaritePredict, zero_division=0, output_dict=True)

	# MARK: TON
	print("\n===== ENTRAÎNEMENT TON =====")

	# Construction du modèle de ton
	tonModel = buildMultilabelModel(
		featureTrain.shape[1], # Nombre de caractéristiques d'entrée (dimensions du TF-IDF)
		tonTrain.shape[1] # Nombre de classes de ton à prédire (ex: Informatif, Ironique, etc.)
	)

	# Entraînement du modèle de ton
	tonModel.fit(
		featureTrain,
		tonTrain,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_split=VALIDATION_SPLIT,
		callbacks=[buildEarlyStopping()],
	)

	# Prédiction et évaluation du modèle de ton
	tonProb = tonModel.predict(featureTest)
	tonPredict = (tonProb >= TON_THRESHOLD).astype(int) # Seuil pour convertir les probabilités en prédictions binaires

	print("\n--- Rapport Ton (TensorFlow) ---")
	print(classification_report(tonTest, tonPredict, target_names=tonMLB.classes_, zero_division=0))
	tonReport = classification_report(tonTest, tonPredict, target_names=tonMLB.classes_, zero_division=0, output_dict=True)

	# MARK: TARGET
	print("\n===== ENTRAÎNEMENT TARGET =====")
	
	# Construction du modèle de cible
	targetModel = buildMulticlassModel(
		featureTrain.shape[1], # Nombre de caractéristiques d'entrée (dimensions du TF-IDF)
		len(targetEncoder.classes_) # Nombre de classes de cible à prédire (ex: International, Politique, etc.)
	)

	# Entraînement du modèle de cible
	targetModel.fit(
		featureTrain,
		targetTrain,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_split=VALIDATION_SPLIT,
		callbacks=[buildEarlyStopping()],
	)

	# Prédiction et évaluation du modèle de cible
	targetProb = targetModel.predict(featureTest)
	targetPredict = targetEncoder.inverse_transform(np.argmax(targetProb, axis=1))
	targetTruth = targetEncoder.inverse_transform(targetTest)

	print("\n--- Rapport Target (TensorFlow) ---")
	print(classification_report(targetTruth, targetPredict, zero_division=0))
	targetReport = classification_report(targetTruth, targetPredict, zero_division=0, output_dict=True)

	# MARK: Sauvegardes des stats/modèles
	SAVEDFOLDER.mkdir(parents=True, exist_ok=True)

	# Sauvegarde des modèles
	polariteModelPath = SAVEDFOLDER / 'polariteModelTensor.keras'
	tonModelPath = SAVEDFOLDER / 'tonModelTensor.keras'
	targetModelPath = SAVEDFOLDER / 'targetModelTensor.keras'

	polariteModel.save(polariteModelPath)
	tonModel.save(tonModelPath)
	targetModel.save(targetModelPath)
	
	joblib.dump(tonMLB, SAVEDFOLDER / 'tonMLBTensor.joblib')
	joblib.dump(polariteEncoder, SAVEDFOLDER / 'polariteLabelTensor.joblib')
	joblib.dump(targetEncoder, SAVEDFOLDER / 'targetLabelTensor.joblib')

	# Sauvegarde des statistiques d'entraînement
	saveTrainingStats(polariteReport, tonReport, targetReport)

	print(f"\nModele polarite sauvegarde dans : {polariteModelPath}")
	print(f"Modele ton sauvegarde dans : {tonModelPath}")
	print(f"Modele target sauvegarde dans : {targetModelPath}")
	print(f"Artefacts labels sauvegardes dans : {SAVEDFOLDER}")

	print("\n===== ENTRAÎNEMENT TERMINÉ =====")

if __name__ == "__main__":
	main()