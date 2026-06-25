# --------------------------------------------------------------
# Nom ...: config.py
# Auteur : Sacha Deliège
# Date ..: Février 2026
# But ...: Configuration et constantes pour les modules d'apprentissage automatique
# .......: chemins vers les fichiers de données, paramètres d'entraînement, etc
# --------------------------------------------------------------

from pathlib import Path

# Racine du projet
PROJECTROOT = Path(__file__).resolve().parents[3]

# Chemin vers les dossiers
DATAFOLDER = PROJECTROOT / "Data" # Dossier principal pour les données, avec sous-dossiers Raw et Processed
CORPUSFOLDER = DATAFOLDER / "Corpus" # Dossier pour les corpus de données (ex : CorpusFrenchTweets1-5M.csv)
PROCESSEDFOLDER = DATAFOLDER / "Processed" # Dossier pour les données traitées (ex : frenchTweetAPI.csv)
RAWFOLDER = DATAFOLDER / "Raw" # Dossier pour les données "brutes" (non formatées, non labélisées)
STATISTICFOLDER = PROJECTROOT / "Statistics" # Dossier pour les statistiques et analyses de corpus (ex : stats_frenchTweetPolitics.json)

TFIDFCORPUSPATH = CORPUSFOLDER / "CorpusFrenchTweets1-5M.csv" # Chemin vers le corpus pour entraîner le modèle TF-IDF (1,5M tweets français)
LABELEDCORPUSPATH = CORPUSFOLDER / "LabeledFrenchTweetPolitics.csv" # Chemin vers le corpus annoté pour entraîner les modèles de polarité et ton (70% de frenchTweetPolitics.csv)
APICORPUSPATH = PROCESSEDFOLDER / "frenchTweetAPI.csv" # Chemin vers le corpus formaté à partir de l'API simulée (30% de frenchTweetPolitics.csv), à normaliser pour plus tard

SAVEDFOLDER = PROJECTROOT / "Modules" / "IAApprentissage" / "ML" / "TrainedModels" # Dossier pour sauvegarder les modèles entraînés
TFIDFMODELPATH = SAVEDFOLDER / "tfidf.joblib" # modèle TF-IDF
POLARITEMODELPATH = SAVEDFOLDER / "polariteModel.joblib" # modèle pour la polarité
TONMODELPATH = SAVEDFOLDER / "tonModel.joblib" # modèle pour le ton
TONMLBMODELPATH = SAVEDFOLDER / "tonMLB.joblib" # MultiLabelBinarizer pour le ton
TARGETMODELPATH = SAVEDFOLDER / "targetModel.joblib" # modèle pour la cible 

PREDICTIONOUTPUTPATH = PROCESSEDFOLDER / "predictions.jsonl" # Chemin pour sauvegarder les prédictions finales au format JSONL

# Paramètres TF-IDF
TFIDFMAXFEATURES = 20000 
TFIDFNGRAMRANGE = (1, 4)
TFIDFMINDF = 3
TFIDFMAXDF = 0.95
TFIDFANALYZER = "char_wb" # Analyseur de caractères