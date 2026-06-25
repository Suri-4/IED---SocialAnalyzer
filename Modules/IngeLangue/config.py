# --------------------------------------------------------------
# Nom ...: config.py
# Auteur : Sacha Deliège
# Date ..: Mars 2026
# But ...: Configuration et chemin pour les modules d'IngeLangue
# --------------------------------------------------------------

from pathlib import Path

# Racine du projet
PROJECTROOT = Path(__file__).resolve().parents[2]

# Chemin vers les dossiers
DATAFOLDER = PROJECTROOT / "Data" # Dossier principal pour les données, avec sous-dossiers Raw et Processed
CORPUSFOLDER = DATAFOLDER / "Corpus" # Dossier pour les corpus de données (ex : CorpusFrenchTweets1-5M.csv)
PROCESSEDFOLDER = DATAFOLDER / "Processed" # Dossier pour les données traitées (ex : frenchTweetAPI.csv)

# Chemin vers le dossier Statistics
STATISTICFOLDER = PROJECTROOT / "Statistics" # Dossier pour les statistiques sur les corpus (nombre de lignes, longueur moyenne des textes, etc.)