from nltk.stem import PorterStemmer , LancasterStemmer
word = "organization"
porter=PorterStemmer()
lancaster=LancasterStemmer()
print(porter.stem(word))
print(lancaster.stem(word))
