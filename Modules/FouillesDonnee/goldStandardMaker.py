# --------------------------------------------------------------
# Nom .........: goldStandardMaker.py
# Auteur ......: Sacha Deliège
# Date ........: Mai 2026
# But .........: Script pour créer un échantillon de gold standard à partir du corpus API
# .............: Ce script prend un échantillon de 200 tweets du corpus API, 
# .............: ajoute des colonnes vides pour les annotations manuelles (polarité, ton, cible, commentaire) et sauvegarde le tout dans un fichier CSV
# Compilation .: python Modules/FouillesDonnee/goldStandardMaker.py
# --------------------------------------------------------------

import pandas as pd
from pathlib import Path

# Chemins d'entrée et de sortie
inputPath = Path("Data/Processed/frenchTweetAPI.csv")
outputPath = Path("Data/GoldStandard/goldStandard.csv")

def main():
	# Vérification de l'existence du fichier d'entrée
	if not inputPath.exists():
		raise FileNotFoundError(f"Fichier introuvable : {inputPath}")

	# Lecture du corpus API
	df = pd.read_csv(inputPath)

	# Vérification des colonnes nécessaires
	if "text" not in df.columns:
		raise ValueError(f"Colonne 'text' absente dans {inputPath}. Colonnes disponibles : {list(df.columns)}")

	# Création de l'échantillon
	goldSample = df.sample(n=200, random_state=42).copy()

	# Colonnes d'annotation manuelle
	goldSample["polariteGold"] = ""
	goldSample["tonGold"] = ""
	goldSample["targetGold"] = ""
	goldSample["commentaireGold"] = ""

	# Colonnes conservées
	goldSample = goldSample[
		[
			"id",
			"text",
			"polariteGold",
			"tonGold",
			"targetGold",
			"commentaireGold",
		]
	]

	# Création du dossier de sortie
	outputPath.parent.mkdir(parents=True, exist_ok=True)

	# Sauvegarde
	goldSample.to_csv(outputPath, index=False, encoding="utf-8")

	print(f"Gold standard créé : {outputPath}")
	print(f"Nombre de tweets sélectionnés : {len(goldSample)}")


if __name__ == "__main__":
	main()