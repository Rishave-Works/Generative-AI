IMDB Sentiment Analysis using Simple RNN
📖 Project Overview

This project implements a Sentiment Analysis system using a Simple Recurrent Neural Network (RNN) trained on the IMDB movie reviews dataset.
The model classifies movie reviews as Positive or Negative based on textual content.

The complete pipeline includes:

Loading and preprocessing the IMDB dataset

Padding sequences for uniform input length

Building and training a Simple RNN model

Applying Early Stopping to avoid overfitting

Evaluating model performance

Visualizing training results

Saving, loading, and testing the trained model on new samples

🧠 Dataset

Dataset: IMDB Movie Reviews

Classes:

1 → Positive Review

0 → Negative Review

Vocabulary Size: Top 1000 most frequent words

Sequence Length: 200 tokens

⚙️ Model Architecture
Layer	Description
Embedding	Converts word indices into 128-dimensional vectors
SimpleRNN	128 neurons with tanh activation
Dense	1 neuron with sigmoid activation for binary classification
🚀 Training Details

Optimizer: Adam

Loss Function: Binary Cross-Entropy

Metrics: Accuracy

Batch Size: 32

Epochs: 20 (with Early Stopping)

Validation Split: 20%

Early stopping monitors validation loss and restores the best model weights.

📊 Model Evaluation

Performance is evaluated using:

Training vs Validation Accuracy

Training vs Validation Loss

Visualizations are generated using Matplotlib

Final evaluation is performed on the test dataset

🧪 Sample Prediction

A trained model predicts sentiment for a sample IMDB review:

Predicted Sentiment: Positive / Negative
Prediction Score: Probability between 0 and 1
Actual Label: Positive / Negative

💾 Model Saving & Loading

Trained model is saved as:

simple_rnn_imdb.h5


The saved model can be reloaded for inference without retraining.

📈 Results

The model successfully learns sequential patterns in text data and achieves good accuracy on the IMDB dataset, demonstrating the effectiveness of RNNs for sentiment analysis tasks.

📂 Project Structure
├── simple_rnn_imdb.h5
├── imdb_rnn_training.py
├── README.md

🔮 Future Improvements

Replace SimpleRNN with LSTM or GRU

Increase vocabulary size

Add Dropout layers to reduce overfitting

Deploy the model using Streamlit or Flask

Add word-level visualization using attention mechanisms

🛠️ Tech Stack
💻 Programming Language

Python

📚 Libraries & Frameworks

TensorFlow / Keras – Model building & training

NumPy – Data handling

Matplotlib – Visualization

🤖 Deep Learning

Recurrent Neural Networks (SimpleRNN)

Embedding Layer

Binary Classification

📊 Dataset

IMDB Movie Reviews Dataset (Keras built-in)