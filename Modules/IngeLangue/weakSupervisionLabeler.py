# --------------------------------------------------------------
# Nom ...: weakSupervisionLabeler.py
# Auteur : Sacha Deliège
# Date ..: Février 2026
# But ...: Labélisation automatique (weak supervision) des corpus CSV
# .......: en utilisant des mots-clés pour les labels de polarité (Favorable, Contre, Défavorable)
# .......: et de ton (Question, Argumentatif, Ironique, Informatif, Autres)
# .......: et génération de statistiques sur la distribution des labels
# --------------------------------------------------------------

import csv
from collections import Counter
import sys
from pathlib import Path
import re
from normalize import normalizeText
import json
import datetime

from config import (
    CORPUSFOLDER,
    STATISTICFOLDER,
)

# Nombre de labels ton à retourner en multi-label (top X par score)
TOP_X_TON_LABELS = 2
TON_MIN_SCORE = 2
TON_RATIO_TO_MAX = 0.5

# MARK: Mots-clés
# Définitions des mots-clés pour chaque label
polariteTerms = {
    "Favorable": [
        "soutenir", "soutien", "approuver", "féliciter", "félicitations",
        "bravo", "applaudir", "applaudissements", "merci", "heureux", 
        "positif", "bon", "bien", "excellent", "meilleur", "optimiste", 
        "victoire", "succès", "réussite", "gagner", "espoirs", "espoir", 
        "fier", "fière", "fierté", "honneur", "respect", "force",
        "liberté", "stabilité", "légitime", "crédible", "solide",
        "populaire", "majorité", "démocrate", "alternatif", "propre",
        "aider", "protéger", "sauver", "rempart", "admire", "admirer", "aimer",
        "bénéfique", "constructif", "solution", "progrès", "enfin",
        "génial", "parfait", "formidable", "super", "top", "pro", "favorable", 
        "bien joué", "bonne idée", "bonne nouvelle", "bonne chance", "chanceux",
        "affirmer", "masterclass", "pépite", "lourd", "incroyable", "mérité"
    ],
    "Défavorable": [
        "contre", "anti", "hostile", "critique", "critiquer", "dénoncer",
        "fraude", "corruption", "mensonge", "triche", "bourrage", "dictateur", "propagande",
        "sanction", "boycott", "attaque", "violence", "répression", "assassin", "tuer",
        "haine", "peur", "danger", "censure", "prison", "mauvais", "négatif", "faible",
        "honte", "honteux", "scandale", "lamentable", "nul", "raté", "minable",
        "problème", "erreur", "injustice", "illégitime", "opposant", "perdre", "perte", "échec", "défaite",
        "triste", "déçu", "déception", "désavantage", "dommage", "désastre", "catastrophe",
        "difficile", "obstacle", "blocage", "défaut", "faillite", "dégradation", "déclin",
        "perdu", "perdue", "perdus", "perdues", "perdant", "perdante", "perdants", "perdantes",
        "insupportable", "marre", "ras-le-bol", "pitié", "pathétique", "naufrage", "fiasco", "débile", "indigne"
    ],
}

tonTerms = {
    "Question": [
        "?", "pourquoi", "comment", "qui", "quoi", "où", "quand", "est-ce", "est-ce que",
        "peut-on", "doit-on", "quel", "quelle", "quels", "quelles", "interrogation", "demander", "demande",
        "questionner", "interroger", "savoir", "vouloir", "pensez-vous", "croyez-vous", "selon vous", 
        "d'après vous", "qu'en pensez-vous", "qu'en est-il", "on en parle", "on en discute", "sérieusement", "vraiment"
    ],
    "Argumentatif": [
        "parce que", "donc", "car", "puisque", "ainsi", "en effet", "à mon avis", "je pense",
        "il faut", "on devrait", "raison", "preuve", "argument", "démontrer", "justifier", "analyse", "opinion",
        "penser", "croire", "défendre", "soutenir", "convaincre", "persuader", "selon", "considérer", "juger",
        "débattre", "discuter", "cela montre", "cela prouve", "il en résulte", "il s'ensuit", "pourtant",
        "il apparaît que", "il ressort que", "il semble que", "il est évident que", "il est clair que", "il est certain que",
        "il est probable que", "il est possible que", "il est nécessaire que", "il est indispensable que",
        "il est important que", "il est essentiel que", "il est primordial que", "il est fondamental que",
        "il est capital que", "il est crucial que", "il est vital que", "il est urgent que", "néanmoins",
        "il est souhaitable que", "il est préférable que", "il est recommandé que", "il est conseillé que",
        "il est suggéré que", "il est proposé que", "il est demandé que", "il est exigé que", "il est requis que",
        "inadmissible", "intolérable", "inacceptable", "réfuter", "contredire", "désaccord", "opposition", "en revanche"
    ],
    "Ironique": [
        "lol", "mdr", "ptdr", "mdrr", "ironie", "ironique", "sarcasme", "sarcastique",
        "ridicule", "absurde", "troll", "trollesque", "moquer", "se moquer", "moqueur", "dérision", "parodie",
        "cynique", "second degré", "surréaliste", "grotesque",
        "quelle surprise", "tiens donc", "sans surprise", "comme par hasard", 
    ],
    "Informatif": [
        "!", "rapport", "rapporter", "annoncer", "déclarer", "selon", "statistique", "chiffre", "résultat",
        "info", "informer", "présenter", "expliquer", "décrire", "observer", "constater",
        "indiquer", "mentionner", "actualité", "news", "journal", "média", "source", "direct", "live",
        "reportage", "témoignage", "événement", "fait", "dire", "affirmer", "détail", "sondage", "moyen",
        "alliance", "coalition", "partenariat", "collaboration", "accord", "traité", "protocole", "fraude",
        "corruption", "investigation", "enquête", "analyse", "étude", "révéler", "divulguer",
        "gouvernement", "administration", "institution", "organisation", "agence", "bureau", "service",
        "loi", "règlement", "réglementation", "politique", "stratégie", "plan", "programme", "initiative", "scrutin",
    ]
}

targetTerms = {
    "Politique": [
        "macron", "gouvernement", "ministre", "premier ministre", "président", "majorité", 
        "politique", "assemblée", "sénat", "élysée", "matignon", "parlement", "législatif", 
        "exécutif", "opposition", "élus", "député", "sénateur", "maire", "collectivité", 
        "préfecture", "préfet", "conseil constitutionnel", "49.3", "décret", "loi", "réforme",
        "lfi", "ps", "pcf", "eelv", "les verts", "nupes", "front populaire", "lr", 
        "republicains", "rassemblement national", "rn", "le pen", "mélenchon", "bardella",
        "renaissance", "modem", "horizon", "reconquête", "extrême droite", "extrême gauche", 
        "centre droit", "centre gauche", "libéral", "souverainiste", "démocratie", "scrutin",
        "élection", "vote", "suffrage", "sondage", "campagne", "mandat",
        "sncf", "greve", "grève", "greves", "grèves", "cheminots", "chomeurs", "chômeurs",
        "fonctionnaires", "retraites", "retraités"
    ],
    "Sécurité": [
        "police", "gendarme", "gendarmerie", "forces de l'ordre", "sécurité", "crs", 
        "policier", "brigade", "bac", "brav-m", "ign", "raid", "gign", "intervention",
        "maintien de l'ordre", "gardien de la paix", "vigile", "sécuriser",
        "prison", "gardien", "cellule", "détenu", "incarcération", "justice", "tribunal", 
        "juge", "procureur", "avocat", "parquet", "condamnation", "plainte", "verdict", 
        "garde à vue", "gav", "interpellation", "délit", "crime", "insécurité",
        "armée", "militaire", "soldat", "guerre", "défense", "état-major", "blindé",
        "marine", "avion de chasse", "général", "conflit armé", "frappe", "bombardement",
        "renseignement", "dgse", "dgsi", "terrorisme", "vigipirate"
    ],
    "Education": [
        "école", "éducation", "université", "lycée", "collège", "classe", "établissement",
        "amphi", "fac", "faculté", "enseignement supérieur", "primaire", "maternelle", 
        "bts", "iut", "grande école", "prépa", "cpge", "internat",
        "professeur", "enseignant", "étudiant", "élève", "apprenti", "stagiaire", 
        "recherche", "chercheur", "doctorant", "rectorat", "académie", "cpe", "instit", 
        "surveillant", "parents d'élèves",
        "bac", "baccalauréat", "diplôme", "brevet", "parcoursup", "apprentissage", 
        "formation", "scolarité", "pédagogie", "programme scolaire", "concours", 
        "bourse", "crous", "cantine", "rythme scolaire", "laïcité"
    ],
    "International": [
        "europe", "ue", "commission européenne", "otan", "onu", "oms", "unesco", 
        "international", "étranger", "mondial", "ocde", "brics", "g7", "g20",
        "ukraine", "russie", "usa", "états-unis", "chine", "allemagne", "royaume-uni", 
        "israël", "palestine", "afrique", "maghreb", "orient", "occident",
        "diplomatie", "frontière", "ambassade", "consulat", "traité", "accord", 
        "sommet", "géopolitique", "expatrié", "migration", "réfugié", "visa", 
        "sanction économique", "embargo", "alliance", "souveraineté",
        "poutine", "vladimir poutine", "vladimir", "russe", "russes", "moscou", "kremlin",
        "crimee", "crimée", "syrie", "trump", "merkel", "navalny", "depardieu",
        "presidentielle russe", "présidentielle russe", "election russe", "élection russe"
    ]
}

# MARK: prepareText()
# Fonction pour préparer le texte avant la recherche de mots-clés (normalisation)
def prepareText(text):
    if text is None: 
        return ""
    return normalizeText(str(text))

# MARK: normalizeTermsDictionary()
# Fonction pour normaliser les termes de chaque label dans les dictionnaires de mots-clés
def normalizeTermsDictionary(termsDict):
    normalized = {}

    for label, terms in termsDict.items(): # Normalisation des termes pour chaque label
        normalized[label] = [] # Initialisation de la liste de termes normalisés pour ce label

        for term in terms: # Normalisation de chaque terme
            normalizedTerm = prepareText(term)
            if normalizedTerm:
                normalized[label].append(normalizedTerm)

    return normalized

# MARK: countKeywordOccurrences()
# Fonction pour compter le nombre d'occurrences d'un mot-clé dans un texte
def countKeywordOccurrences(text, keyword):
    if not keyword: # Si pas de mot-clé, retourner 0
        return 0

    # Pour les signes comme ? ou ...
    if not any(char.isalnum() for char in keyword):
        return text.count(keyword)

    # Pour les expressions ou mots simples : bornes lexicales
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return len(re.findall(pattern, text))

# MARK: calcScore()
# Fonction pour calculer les scores de chaque label en fonction des mots-clés présents dans le texte
def calcScore(text, termsByLabel):
    scores = {} # Dictionnaire pour stocker les scores de chaque label

    for label, terms in termsByLabel.items(): # Calcul des scores pour chaque label
        score = 0

        for term in terms: # Comptage des occurrences de chaque terme
            score += countKeywordOccurrences(text, term)

        if score > 0:
            scores[label] = score
    return scores

# MARK: chooseBestLabel()
# Fonction pour choisir le label avec le score le plus élevé, en cas d'égalité utiliser l'ordre de priorité défini
def chooseBestLabel(scores, priorityOrder, defaultLabel):
    if not scores: # Si aucun score n'est positif, retourner le label par défaut ("Neutre" pour la polarité, "Autre" pour le ton et la cible)
        return defaultLabel

    maxScore = max(scores.values()) # Trouver le score maximum parmi les labels
    candidates = [label for label, score in scores.items() if score == maxScore] # Trouver les labels qui ont ce score maximum

    for label in priorityOrder: # Parcourir l'ordre de priorité pour trouver le premier label qui est dans les candidats
        if label in candidates: # Si ce label est parmi les candidats, le retourner
            return label

    return candidates[0] # Si aucun des candidats n'est dans l'ordre de priorité (peu probable), retourner le premier candidat trouvé

# MARK: chooseTopTons()
# Fonction pour choisir les labels de ton à retourner en multi-label (top X par score, avec un score minimum et un ratio par rapport au score maximum)
def chooseTopTons(scores):
    if not scores: # Si aucun score n'est positif, retourner "Autres"
        return "Autres"

    maxScore = max(scores.values()) # Trouver le score maximum parmi les labels de ton
    tonPriorityIndex = {label: index for index, label in enumerate(tonPriority)} # Dictionnaire pour l'index de priorité de chaque label de ton
    selectedLabels = [] # Liste pour stocker les labels de ton sélectionnés

    for label in tonPriority: # Pour chaque label de ton dans l'ordre de priorité, vérifier s'il remplit les conditions pour être sélectionné
        score = scores.get(label, 0) # Récupérer le score de ce label, ou 0 s'il n'est pas présent dans les scores

        if score >= TON_MIN_SCORE and score >= maxScore * TON_RATIO_TO_MAX: # Vérifier si le score remplit les conditions pour être sélectionné
            selectedLabels.append(label) # Ajouter ce label à la liste des labels sélectionnés

    if not selectedLabels: # Si aucun label ne remplit les conditions, sélectionner le label avec le score maximum (même s'il est en dessous du seuil)
        selectedLabels = [max(scores, key=scores.get)] 

    # Trier les labels sélectionnés en fonction de leur score (du plus élevé au plus bas) et de leur ordre de priorité
    selectedLabels.sort(key=lambda label: (-scores[label], tonPriorityIndex.get(label, len(tonPriority))))
    
    return ";".join(selectedLabels[:TOP_X_TON_LABELS])

# Normalisation des termes de chaque label dans les dictionnaires de mots-clés
polariteTermsNormalized = normalizeTermsDictionary(polariteTerms)
tonTermsNormalized = normalizeTermsDictionary(tonTerms)
targetTermsNormalized = normalizeTermsDictionary(targetTerms)

# Ordre de priorité des labels en cas d'égalité de score 
polaritePriority = ["Défavorable", "Favorable"]
tonPriority = ["Argumentatif", "Question", "Informatif", "Ironique"]
targetPriority = ["International", "Sécurité", "Education", "Politique"]

# MARK: labelPolarite()
# Fonctions pour labéliser le texte
def labelPolarite(text):
    textPrep = prepareText(text) # Préparer le texte (normalisation)
    scores = calcScore(textPrep, polariteTermsNormalized) # Calculer les scores de chaque label de polarité en fonction des mots-clés présents dans le texte
    # Retourner le label de polarité avec le score le plus élevé ou "Neutre" si aucun score n'est positif
    return chooseBestLabel(scores, polaritePriority, "Neutre")
    
# MARK: labelTon()
# Fonctions pour labéliser le ton
def labelTon(text):
    textPrep = prepareText(text)
    scores = calcScore(textPrep, tonTermsNormalized) 
    return chooseTopTons(scores)

# MARK: labelTarget()
def labelTarget(text):
    textPrep = prepareText(text)
    scores = calcScore(textPrep, targetTermsNormalized) 
    return chooseBestLabel(scores, targetPriority, "Autre")

# MARK: percentage()
# Calcule un pourcentage
def percentage(count, total):
    if total <= 0:
        return 0

    valuePercentage = (count/total)*100 # Calcul du pourcentage
    return valuePercentage

# MARK: buildDistributionStats()
# Fonction pour transformer un compteur en statistiques exploitables
def buildDistributionStats(counter):
    total = sum(counter.values()) # Nombre total d'occurrences de tous les labels
    distribution = {} # Dictionnaire pour stocker la distribution des labels avec leur nombre et pourcentage

    for label, count in counter.items(): # Pour chaque label, calculer le pourcentage par rapport au total
        labelPercentage = percentage(count, total)

        distribution[label] = {"count": count, "percentage": labelPercentage}

    return {"total": total, "distribution": distribution}

# MARK: printDistributionStats()
# Fonction pour afficher les statistiques dans le terminal
def printDistributionStats(title, counter):
    total = sum(counter.values())

    print(f"\n{title} :")

    for label, count in counter.items():
        labelPercentage = percentage(count, total)

        print(f"    {label} : {count} ({labelPercentage:.2f}%)")

# MARK: saveWeakSupervisionStats()
# Fonction pour afficher et sauvegarder les statistiques de weak supervision
def saveWeakSupervisionStats(polariteCount, tonCount, targetCount, outputFilePath):
    print("\n=== Statistiques weak supervision ===")

    # Affichage des statistiques dans le terminal
    printDistributionStats("Polarité", polariteCount)
    printDistributionStats("Ton", tonCount)
    printDistributionStats("Target", targetCount)

    # Génération du rapport de statistiques au format JSON
    report = {
        "file": outputFilePath.name,
        "generatedAt": datetime.datetime.now().isoformat(),
        "polarite": buildDistributionStats(polariteCount),
        "ton": buildDistributionStats(tonCount),
        "target": buildDistributionStats(targetCount)
    }

    STATISTICFOLDER.mkdir(parents=True, exist_ok=True)
    reportPath = STATISTICFOLDER / "weakSupervisionStats.json"

    # Sauvegarde du rapport de statistiques au format JSON
    with open(reportPath, "w", encoding="utf-8") as fOUT:
        json.dump(report, fOUT, ensure_ascii=False, indent=4)

    print(f"\nStatistiques weak supervision sauvegardées dans : {reportPath}")

# MARK: main()
def main():
    inputFilePath = CORPUSFOLDER / "CorpusFrenchTweetPolitics.csv"
    outputFilePath = CORPUSFOLDER / "LabeledFrenchTweetPolitics.csv"
    # Vérification de l'existence du fichier d'entrée
    if not inputFilePath.exists():
        raise FileNotFoundError(f"Fichier d'entrée non trouvé : {inputFilePath}")
    
    # Compteurs pour les statistiques de distribution des labels
    polariteCount = Counter()
    tonCount = Counter()
    targetCount = Counter()

    # Lecture et annotation du corpus
    with open(inputFilePath, "r", newline="", encoding="utf-8") as fIN, open(outputFilePath, "w", newline="", encoding="utf-8") as fOUT:
        reader = csv.DictReader(fIN)
        fieldnames = reader.fieldnames
        if "polarite" not in fieldnames:
            fieldnames.append("polarite")
        if "ton" not in fieldnames:
            fieldnames.append("ton")
        if "target" not in fieldnames:
            fieldnames.append("target")

        writer = csv.DictWriter(fOUT, fieldnames=fieldnames) # Écriture dans le fichier de sortie
        writer.writeheader()

        # Parcours des lignes du corpus
        for row in reader:
            text = row.get("text", "")

            # Labélisation du texte
            polarite = labelPolarite(text)
            ton = labelTon(text)
            target = labelTarget(text)

            # Mise à jour de la ligne avec les labels trouvés
            row["polarite"] = polarite
            row["ton"] = ton
            row["target"] = target

            # Mise à jour des compteurs pour les statistiques
            polariteCount[polarite] += 1
            tonCount[ton] += 1
            targetCount[target] += 1

            writer.writerow(row) # Écriture de la ligne annotée dans le fichier de sortie

    # Affichage et sauvegarde des statistiques de weak supervision
    saveWeakSupervisionStats(polariteCount, tonCount, targetCount, outputFilePath)

    print("\nCorpus annoté sauvegardé dans :", outputFilePath)

if __name__ == "__main__":
    main()