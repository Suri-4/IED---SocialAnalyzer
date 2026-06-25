# --------------------------------------------------------------
# Nom ...: formatCorpus.py
# Auteur : Sacha Deliège
# Date ..: Février 2026
# But ...: formatage des corpus CSV pour analyse sociale 
# .......: retrait des colonnes inutiles, réordonnancement des IDs, ajout de colonnes vides pour labels ton et polarité
# .......: découpage 70/30 pour frenchTweetPolitics.csv afin d'obtenir un corpus formaté et un corpus brut pour
# .......: respectivement la weak supervision et la prédiction finale sur modèle déjà entraîné
# --------------------------------------------------------------

import csv
import json
import datetime
from normalize import normalizeText

from config import (
	CORPUSFOLDER,
	PROCESSEDFOLDER,
	STATISTICFOLDER,
)

# MARK: formatCorpusCSV()
# Fonction pour formater un corpus CSV afin d'avoir les colonnes id, text, polarite, ton
def formatCorpusCSV(inputPath, outputPath):
	rows = [] # Liste pour stocker les lignes
	with open(inputPath, 'r', newline='', encoding='utf-8') as fIN:
		reader = csv.DictReader(fIN) # Lire le CSV
		for row in reader:
			rows.append({
				'text': normalizeText(row.get('text')),
			})
			
	# Réordonner les IDs (utile pour frenchTweetPolitics)
	for idx, row in enumerate(rows):
		row['id'] = idx
		row['polarite'] = ''
		row['ton'] = ''
		row['target'] = ''
	
	# Écrire le fichier formaté
	with open(outputPath, 'w', newline='', encoding='utf-8') as fOUT:
		fieldnames = ['id', 'text', 'polarite', 'ton', 'target']
		csvFormated = csv.DictWriter(fOUT, fieldnames=fieldnames)
		csvFormated.writeheader()
		for row in rows:
			csvFormated.writerow({field: row.get(field, '') for field in fieldnames})

# MARK: splitCorpus()
# Fonction simple pour séparer frenchTweetPolitics.csv en 70/30
def splitCorpus(inputPath, outPath70, outPath30):
	with open(inputPath, 'r', newline='', encoding='utf-8') as fIN: # Lecture du fichier d'entrée
		reader = list(csv.DictReader(fIN)) # Convertir en liste pour accès par index

	total = len(reader) # Nombre total de lignes
	splitIDX = int(0.7 * total) # 70% premiers pour train, 30% derniers pour test (split stable, non aléatoire)
	rows70 = []
	for i, row in enumerate(reader[:splitIDX]):
		rows70.append({
			'id': i,
			'text': normalizeText(row.get('text')),
			'polarite': '',
			'ton': '',
			'target': ''
		})
		
	# Écrire le fichier formaté
	with open(outPath70, 'w', newline='', encoding='utf-8') as fOUT:
		fieldnames = ['id', 'text', 'polarite', 'ton', 'target']
		csvFormated = csv.DictWriter(fOUT, fieldnames=fieldnames)
		csvFormated.writeheader()
		for row in rows70:
			csvFormated.writerow(row)

	# 30% restant pour test (non formaté, API simulée)
	rows30 = []
	for i, row in enumerate(reader[splitIDX:]):
		rows30.append({'id': i, 'text': normalizeText(row.get('text'))}) # Écrire le fichier non formaté
	with open(outPath30, 'w', newline='', encoding='utf-8') as fOUT:
		fieldnames = ['id', 'text']
		csvFormated = csv.DictWriter(fOUT, fieldnames=fieldnames)
		csvFormated.writeheader()
		for row in rows30:
			csvFormated.writerow(row)

# MARK: saveStatCorpus()
def saveStatCorpus(file):
	filePath = CORPUSFOLDER / file
	if not filePath.exists(): # Vérification de l'existence du fichier
		raise FileNotFoundError(f"Fichier non trouvé : {filePath}")
	
	# Statistiques
	with open(filePath, 'r', newline='', encoding='utf-8') as fIN:
		reader = csv.DictReader(fIN)
		# Initialisation des compteurs et variables pour les statistiques
		nbrLines, TextLength, totalWords, emptyTexts = 0, 0, 0, 0
		minLength, maxLength, minWord, maxWord = None, None, None, None
		vocab = set() # Ensemble pour stocker le vocabulaire unique

		for row in reader: # Parcours chaque ligne du fichier CSV pour calculer les statistiques
			nbrLines += 1
			text = row.get('text', '')
			if text is None:
				text = ''
			words = text.split() # Séparer le texte en mots
			wordCount = len(words)

			# Longueur en caractères
			TextLength += len(text)
			if minLength is None or maxLength is None: # Initialisation des longueurs min et max en taille de texte
				minLength, maxLength = len(text), len(text)
			else:
				minLength = min(minLength, len(text))
				maxLength = max(maxLength, len(text))

			# Longueur en mots
			totalWords += wordCount
			if minWord is None or maxWord is None: # Initialisation des longueurs min et max en mots
				minWord, maxWord = wordCount, wordCount
			else:
				minWord = min(minWord, wordCount)
				maxWord = max(maxWord, wordCount)

			# Textes vides
			if not text.strip(): # Vérification si le texte est vide ou ne contient que des espaces
				emptyTexts += 1

			# Vocabulaire unique
			vocab.update(words) # Ajouter les mots du texte à l'ensemble de vocabulaire unique

	# Calcul des statistiques
	averageLength = TextLength/nbrLines
	averageWordsPerText = totalWords/nbrLines
	emptyTextsPercentage = round(emptyTexts/nbrLines*100, 2)

	# Rapport de statistiques
	report = {
		'file':                  filePath.name,
		'generatedAt':           datetime.datetime.now().isoformat(),
		'totalLines':            nbrLines,
		'averageTextLength':     averageLength,
		'minTextLength':         minLength,
		'maxTextLength':         maxLength,
		'minWord':               minWord,
		'maxWord':               maxWord,
		'averageWordsPerText':   averageWordsPerText,
		'emptyTexts':            emptyTexts,
		'emptyTextsPercentage':  emptyTextsPercentage,
		'vocabSize':             len(vocab),
	}

	statsFolder = STATISTICFOLDER
	statsFolder.mkdir(exist_ok=True)
	reportPath = statsFolder / f"{filePath.stem}Stats.json"
	with open(reportPath, 'w', encoding='utf-8') as fOUT:
		json.dump(report, fOUT, ensure_ascii=False, indent=4)
	print(f"Statistiques sauvegardées dans : {reportPath}")

if __name__ == "__main__":
	folderPath = CORPUSFOLDER

	# Liste des fichiers à formater (input, output)
	files = [
		('frenchTweetPolitics.csv', 'CorpusFrenchTweetPolitics.csv'),
		('frenchTweets1-5M.csv', 'CorpusFrenchTweets1-5M.csv'),
	]

	# Formatage des corpus
	for fileIn, fileOut in files:
		inPath = CORPUSFOLDER / fileIn
		outPath = CORPUSFOLDER / fileOut
		print(f'Normalisation de {fileIn}...') 

		try: # Appel de la fonction de formatage pour chaque fichier
			formatCorpusCSV(inPath, outPath)
			print(f'Fichier normalisé écrit : {outPath}')
			saveStatCorpus(inPath) # Statistiques sur le fichier d'entrée (non formaté)
			saveStatCorpus(outPath) # Statistiques sur le fichier de sortie (formaté)
		except Exception as e: # Gestion des erreurs lors du formatage
			raise Exception(f'Erreur lors du formatage de {fileIn} : {e}')

	# Split 70/30 pour frenchTweetPolitics.csv
	inPath = CORPUSFOLDER / 'frenchTweetPolitics.csv' # Chemin d'entrée pour le split
	outPath70 = CORPUSFOLDER / 'CorpusFrenchTweetPolitics.csv' # Chemin de sortie pour les 70% formatés

	# outPath30 = os.path.join(folderProcessedPath, 'frenchTweetAPI.csv')
	outPath30 = PROCESSEDFOLDER / 'frenchTweetAPI.csv' # Chemin de sortie pour les 30% non formatés (API simulée)
	print('Découpage 70/30 de frenchTweetPolitics.csv...')
	try:
		splitCorpus(inPath, outPath70, outPath30)
		saveStatCorpus(outPath70)
		saveStatCorpus(outPath30)
		print(f'70% formaté écrit dans : {outPath70}')
		print(f'30% formaté écrit dans : {outPath30}')
	except Exception as e:
		raise Exception(f'Erreur lors du découpage 70/30 : {e}')

