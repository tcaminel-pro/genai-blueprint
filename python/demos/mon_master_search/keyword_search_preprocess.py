import spacy

print("load Spacy model")
nlp_fr = spacy.load("fr_core_news_sm")
stop_words = nlp_fr.Defaults.stop_words
stop_words.update({",", ";", "(", ")", ":", "[", "]"})


def preprocess_text(text) -> list[str]:
    r = [token.lemma_.lower() for token in nlp_fr(text) if token not in stop_words]
    debug(r)
    return r
