import numpy as np

from tensorflow.keras.datasets import imdb

max_features=1000

imdb.load_data(num_words=max_features)

(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_features)

print(f"Training data shape : {len(X_train)},Training labels shape:{len(y_train)}")
print(f"Testing data shape : {len(X_test)},Testing labels shape:{len(y_test)}")

X=np.concatenate((X_train,X_test),axis=0)
y=np.concatenate((y_train,y_test),axis=0)

split_index=int(0.8*len(X))

X_train=X[:split_index]
y_train=y[:split_index]

X_test=X[split_index:]
y_test=y[split_index:]

print("Training samples:",len(X_train))
print("Testing samples:",len(X_test))

print(X_train[0])
print(y_train[0])

from tensorflow.keras.preprocessing import sequence

max_len=200

X_train=sequence.pad_sequences(X_train,maxlen=max_len)

X_test=sequence.pad_sequences(X_test,maxlen=max_len)

print(X_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

model=Sequential()

model.add(Embedding(input_dim=max_features,output_dim=128))

model.add(SimpleRNN(128,activation='tanh'))

model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

earlystopping=EarlyStopping(
    monitor='Val_loss',
    patience=5,
    restore_best_weights=True
)

history=model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)

import matplotlib.pyplot as plt

accuracy=history.history['accuracy']
val_accuracy=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(accuracy)+1)

plt.figure(figsize=(7,4))

plt.plot(epochs,accuracy,color='blue',marker='o',label='Training Accuracy')
plt.plot(epochs,val_accuracy,color='green',marker='s',label='Validation Accuracy')


plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))

plt.plot(epochs,loss,color='red',marker='o',label='Training Loss')
plt.plot(epochs,val_loss,color='blue',marker='s',label='Validation Loss')


plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

model.save("simple_rnn_imdb.h5")

from tensorflow.keras.models import load_model

model=model.load_model("simple_rnn_imdb.h5")
model.sumarry()

test_loss,test_accuracy=model.evaluate(X_test,y_test)

print(f"Test Loss:{test_loss}")
print(f"Test Accuracy:{test_accuracy}")

sample_review=X_test[1]

print(sample_review)

print(len(sample_review))

len(sample_review.reshape(1,-1))

prediction=model.predict(sample_review.reshape(1,-1))

print(prediction)

sentiment="Positive" if prediction[0][0] >0.5 else "Negative"
print("Predicted Sentiment:",sentiment)
print("Prediction Score:",prediction[0][0])

print("Actual label:","Positive" if y_test[0]==1 else "Negative")








