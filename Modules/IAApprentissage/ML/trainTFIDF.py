# --------------------------------------------------------------
# Nom ...: trainTFIDF.py
# Auteur : Sacha Deliège
# Date ..: Février 2026
# But ...: Entraîner et sauvegarder un modèle TF-IDF sur le corpus massif de tweets français (1,5M) 
# .......: pour l'utiliser comme base de représentation dans les modèles d'apprentissage supervisé
# --------------------------------------------------------------

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Import des constantes de configuration pour le chemin du corpus et les paramètres du TF-IDF
from config import (
    TFIDFCORPUSPATH,
    TFIDFMODELPATH,
    TFIDFMAXFEATURES,
    TFIDFNGRAMRANGE,
    TFIDFMINDF,
    TFIDFMAXDF,
    TFIDFANALYZER
)

# MARK: main()
def main():
    if not TFIDFCORPUSPATH.exists(): # Vérifier que le corpus existe
        raise FileNotFoundError(
            f"Corpus introuvable : {TFIDFCORPUSPATH}"
        )

    corpus = pd.read_csv(TFIDFCORPUSPATH) # Charger le corpus

    if 'text' not in corpus.columns: # Vérification
        raise ValueError(
            f"Colonne 'text' absente.\nListe des colonnes disponibles : {list(corpus.columns)}"
        )

    texts = corpus['text'].fillna("").astype(str) # éviter les NaN et convertir en string (Err pandas sinon)

    print(f"Nombre de textes utilisés : {len(texts)}")

    vectorizer = TfidfVectorizer( # TF-IDF avec les paramètres définis dans config.py
        lowercase=False, # déjà normalisé dans formatCorpus.py
        max_features=TFIDFMAXFEATURES,
        ngram_range=TFIDFNGRAMRANGE, # unigrammes, bigrammes pour capturer les expressions courantes
        min_df=TFIDFMINDF, 
        max_df=TFIDFMAXDF,
        sublinear_tf=True, # Réduit l'impact des termes très fréquents 
        analyzer=TFIDFANALYZER, # Analyseur de caractères pour capturer les variations orthographiques et les hashtags
    )

    print("Apprentissage du vocabulaire TF-IDF")
    vectorizer.fit(texts) # Apprendre le vocabulaire à partir des textes du corpus
    joblib.dump(vectorizer, TFIDFMODELPATH) # Sauvegarder le modèle entraîné
    print(f"Modèle sauvegardé dans : {TFIDFMODELPATH}")


if __name__ == "__main__":
    main()
