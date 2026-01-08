import spacy
nlp=spacy.load("en_core_web_sm")
doc=nlp("spaCy is very powerful")
#Accessing tokens
tokens_text=[token.text for token in doc]
tokens_obj=[token for token in doc]
print(tokens_text)
print(tokens_obj)