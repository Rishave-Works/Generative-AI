from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
corpus = ["text one", "text two", "text three"]
vector = TfidfVectorizer()
bow=CountVectorizer()
X = vector.fit_transform(corpus)
Y = bow.fit_transform(corpus)
print(X.toarray())
print(Y.toarray())
print(vector.get_feature_names_out())