# --------------------------------------------------------------
# Nom ...: runSocialAnalyzer.py
# Auteur : Sacha Deliège
# Date ..: Février 2026
# But ...: Script principal pour exécuter les différentes étapes de SocialAnalyzer
# .......: formatage du corpus, labélisation par weak supervision, entraînement du modèle, prédiction sur le corpus API, etc
# .......: 
# --------------------------------------------------------------

import subprocess
import sys
import argparse

# Fonction pour exécuter une étape
def runStep(moduleName, command):
    print(f"\n=== {moduleName} ===\n")
    try:
        subprocess.run(command, check=True) # Exécuter la commande et vérifier le succès
        print(f"[OK] {moduleName}") 
    except subprocess.CalledProcessError: # Gestion des erreurs lors de l'exécution de la commande
        print(f"[ERREUR] {moduleName}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="runSocialAnalyzer.py",
        description="Script principal pour exécuter les différentes étapes",
    )
    parser.add_argument(
        "--skipFormatting", "-sf",
        help="Passer l'étape de formatage du corpus",
        action="store_true"
    )
    parser.add_argument(
        "--skipLabeling", "-sl",
        help="Passer l'étape de labélisation par weak supervision",
        action="store_true"
    )
    parser.add_argument(
        "--skipTFIDF", "-st",
        help="Passer l'étape d'entraînement du modèle TF-IDF",
        action="store_true"
    )
    parser.add_argument(
        "--skipLabelTraining", "-slt",
        help="Passer l'étape d'entraînement des modèles de polarité et ton",
        action="store_true"
    )
    parser.add_argument(
        "--skipPrediction", "-sp",
        help="Passer l'étape de prédiction sur le corpus API",
        action="store_true"
    )
    args = parser.parse_args() # Analyse des arguments pour permettre de sauter certaines étapes si nécessaire

    print("===== Lancement de SocialAnalyzer =====")

    # Étape 1 : Formatage du corpus
    if not args.skipFormatting:
        runStep("Prétraitement - Ingenierie des Langues", ["python", "Modules/IngeLangue/formatCorpus.py"])

    # Étape 2 : Labélisation par weak supervision
    if not args.skipLabeling:
        runStep("Labélisation - Weak Supervision", ["python", "Modules/IngeLangue/weakSupervisionLabeler.py"])
    
    # Étape 3 : Entraînement du modèle 
    if not args.skipTFIDF:
        runStep("Entraînement du modèle TF-IDF", ["python", "Modules/IAApprentissage/ML/trainTFIDF.py"])

    if not args.skipLabelTraining:
        runStep("Entraînement des modèles de polarité et ton", ["python", "Modules/IAApprentissage/ML/trainTensorLabel.py"])

    # Étape 4 : Prédiction sur le corpus API
    if not args.skipPrediction:
        runStep("Prédiction sur le corpus API", ["python", "Modules/IAApprentissage/ML/predictionTensor.py"])
