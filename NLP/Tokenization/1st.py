import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
#nltk.download('punkt')
#nltk.download('punkt_tab')
text= "AI is amazing. It helps humans."
print("word tokenize")
print(word_tokenize(text))
print("Sentence tokenize")
print(sent_tokenize(text))