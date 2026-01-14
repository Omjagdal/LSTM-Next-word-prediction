# 📘 LSTM Next Word Prediction (Google Colab)

## 📌 Project Overview
This project implements a Next Word Prediction system using LSTM in Google Colab. The model learns from a small text dataset and predicts the next most likely word for a given input sequence.

The project is suitable for:
- NLP beginners
- Academic mini / final-year projects
- Demonstrations using Google Colab (no local setup required)

## 🧠 Problem Statement 
Given a sequence of words, predict the next word using a deep learning model.

**Example:**
```
Input  : "machine learning is"
Output : "powerful"
```

## ☁️ Why Google Colab
- No installation required
- Free GPU/CPU support
- Easy sharing and reproducibility
- Ideal for student projects

## 📊 Dataset
**Format:** `.txt`  
**Size:** Small (5 KB – 200 KB recommended)  
**Content examples:**
- Short stories
- Articles
- Notes
- Scripts

**Sample dataset (text.txt):**
```
Deep learning is a subset of machine learning.
Machine learning is a part of artificial intelligence.
```

## ⚙️ Libraries Used
All libraries are pre-installed in Colab.
```python
import tensorflow as tf
import numpy as np
import pickle
```

## 🏗️ Model Architecture
The model consists of:
- Embedding Layer
- LSTM Layer
- Dense Output Layer (Softmax)

**Architecture flow:**
```
Text → Tokenizer → Embedding → LSTM → Dense → Next Word
```

## 🚀 Training the Model (Colab)
1. Upload `text.txt` to Colab
2. Run the training notebook
3. Train the model:
```python
model.fit(X, y, epochs=100, verbose=1)
```

**During training:**
- Loss decreases gradually
- Model learns word sequences
- Trained model is saved as `lstm_model.h5`

## 🔮 Predicting the Next Word
**Example usage:**
```python
seed_text = "deep learning is"
next_word = predict_next_word(seed_text)
print(next_word)
```

**Output:**
```
deep learning is powerful
```

## 📈 Evaluation
Training loss is the main metric. Accuracy improves with:
- More data
- More epochs
- Cleaner text

⚠️ **Note:** Small dataset = limited vocabulary (expected behavior)

## 🧪 Experiments You Can Try
- Increase epochs (50 → 200)
- Change sequence length
- Add another LSTM layer
- Use larger text file

## 🎓 Applications
- Text auto-completion
- Chatbot basics
- NLP learning
- Academic demonstrations

## ❗ Limitations
- Works best on small, clean datasets
- Grammar may not always be perfect
- Not comparable to large language models

## ✅ Conclusion
This project demonstrates how LSTM networks can learn language patterns and perform next word prediction using a small dataset in Google Colab. It is simple, educational, and ideal for student projects.

## 📂 Project Structure
```
lstm-next-word-prediction/
├── text.txt
├── lstm_next_word_prediction.ipynb
├── lstm_model.h5
├── tokenizer.pickle
└── README.md
```

## 🚀 Getting Started

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com/)

### Step 2: Upload Dataset
Upload your `text.txt` file to Colab

### Step 3: Run the Notebook
Execute all cells in sequence

### Step 4: Make Predictions
Use the trained model to predict next words

## 📝 License
MIT License

## 👤 Author
Your Name - OM JAGDALE

## 🙏 Acknowledgments
- TensorFlow Documentation
- Keras LSTM Tutorials
- Google Colab Community
