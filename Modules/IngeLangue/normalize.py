# --------------------------------------------------------------
# Nom ...: normalize.py
# Auteur : Sacha Deliège
# Date ..: Février 2026
# But ...: Normalisation et nettoyage de textes
# .......: Suppression du bruit (URLs, mentions, emojis, etc.)
# .......: et uniformisation des textes pour une meilleure analyse sociale
# --------------------------------------------------------------

import re
import emoji
import unicodedata

# MARK: removeAccents()
# Fonction pour retirer les accents unicode d'un texte
def removeAccents(text):
	if text is None:
		return ''
	text = unicodedata.normalize('NFD', text)
	return ''.join(c for c in text if not unicodedata.combining(c))

# MARK: removeEmojis()
# Fonction pour retirer les émojis d'un texte
def removeEmojis(text):
    return ''.join(char for char in text if char not in emoji.EMOJI_DATA)

# MARK: normalizeText()
# Fonction pour normaliser un texte (nettoyage et uniformisation)
# Met en minuscules, normalise les apostrophes et guillemets, retire les URLs, remplace les mentions utilisateur par @USER, 
# retire les émojis, réduit la ponctuation répétée, retire les espaces multiples
def normalizeText(text):
	if text is None:
		return ''

	# Mettre en minuscules
	text = text.lower()

	# Normaliser les apostrophes et guillemets
	text = text.replace('’', "'").replace('‘', "'")
	text = text.replace('“', '"').replace('”', '"')

	# Retirer les URLs
	text = re.sub(r'https?://\S+', '', text) # Retire les URLs commençant par http:// ou https://
	text = re.sub(r'www\S+', '', text) # Retire les URLs commençant par www.

	# Remplacer les mentions utilisateur (@username) par un token neutre
	text = re.sub(r'@\w+', '@USER', text)

	# Retirer les accents unicode
	text = removeAccents(text)

	# Retirer les émojis
	text = removeEmojis(text)

	# Réduire la ponctuation répétée (!!! -> !, ??? -> ?)
	text = re.sub(r'([!?.])\1+', r'\1', text)

	# Retirer les espaces multiples
	text = re.sub(r'\s+', ' ', text).strip()

	return text
