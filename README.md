# Online Hate Speech Detection in Portuguese (BiLSTM Baseline)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![NLP](https://img.shields.io/badge/NLP-Natural_Language_Processing-blueviolet.svg)
![Academic Project](https://img.shields.io/badge/Academic_Project-Research-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 1. Project Overview

This project focuses on the challenging task of **detecting online hate speech in Portuguese**. It represents a foundational step in my academic research towards developing robust and nuanced models for combating harmful content on the internet.

This repository specifically contains a **Bidirectional Long Short-Term Memory (BiLSTM) model**, serving as a comprehensive baseline and a deep dive into recurrent neural network architectures. The insights gained from this implementation will be crucial for the next phase of my research, which involves developing a hybrid BiLSTM-Transformer model.

Test this model here: https://huggingface.co/spaces/carpenterbb/curupira-bilstm-baseline-app
---

## 2. The Problem: Online Hate Speech in Portuguese

Online hate speech is a growing global concern, contributing to harassment, discrimination, and radicalization. In Brazil and other Portuguese-speaking countries, the problem is exacerbated by the linguistic nuances, cultural contexts, and informal nature of online communication, making automated detection particularly complex.

**Challenges in Portuguese Hate Speech Detection:**

* **Contextual Nuance:** Words and phrases can be offensive in one context but harmless in another.
* **Slang & Informal Language:** Online communication often involves heavy use of slang, abbreviations, and emojis.
* **Syntactic Complexity:** Portuguese, like other Romance languages, has rich morphology and flexible sentence structures.
* **Class Imbalance:** Hate speech instances are typically a minority compared to general online discourse, posing a significant challenge for model training.

This project aims to contribute to the development of effective tools to identify such content, fostering safer online environments.

---

## 3. Academic Research Context

This project is an integral part of my ongoing academic research focused on **advanced Natural Language Processing (NLP) techniques for hate speech detection**. My long-term goal is to explore and implement **hybrid deep learning architectures combining BiLSTMs and Transformers** to leverage the strengths of both: BiLSTMs for capturing sequential dependencies and Transformers for understanding long-range contextual relationships.

This BiLSTM-only implementation is a critical step to:

1.  **Understand Recurrent Architectures:** Gain a deep practical understanding of BiLSTM networks, including their training dynamics, hyperparameter tuning, and performance characteristics in the context of sequence classification.
2.  **Establish a Performance Baseline:** Provide a strong, well-documented baseline against which more complex future models (like the BiLSTM-Transformer hybrid) can be compared and evaluated.
3.  **Refine Data Preprocessing for Portuguese:** Develop and optimize robust text preprocessing pipelines specifically for online Portuguese content.

---

## 4. Methodology & Model Architecture

This project employs a deep learning approach using a Bidirectional Long Short-Term Memory (BiLSTM) network.

### 4.1. Data Preparation


1.  **Dataset Acquisition:** Hugging Face
2.  **Text Preprocessing:**
    * Lowercasing.
    * Removal of special characters, URLs, and mentions.
    * Tokenization using (e.g., `tf.keras.preprocessing.text.Tokenizer` or NLTK).
    * Padding sequences to a uniform length.
3.  **Vocabulary Creation:** Building a vocabulary from the training data.
4.  **Embedding Layer:** Utilizing a pre-trained Word2Vec/FastText embedding for Portuguese
5.  **Train-Validation-Test Split:** Splitting the data into training, validation, and test sets to evaluate generalization capabilities.

### 4.2. BiLSTM Model Architecture

The core of the model is a BiLSTM layer.
Input Layer (Variable Sequence Length) ↓ Embedding Layer (Word Embeddings) ↓ Bidirectional LSTM Layer (captures context from both directions) ↓ Dropout Layer (for regularization) ↓ Dense Layer (ReLU activation) ↓ Dropout Layer (for regularization) ↓ Output Layer (Sigmoid activation for binary classification)

* **Embedding Layer:** Converts numerical tokens into dense vector representations.
* **BiLSTM Layer:** Processes the sequence in both forward and backward directions, allowing it to capture long-range dependencies and context from both past and future words in a sentence. This is crucial for understanding the subtleties of hate speech.
* **Dropout:** Applied after the BiLSTM and Dense layers to prevent overfitting.
* **Dense Layers:** Standard fully connected neural network layers for feature transformation.
* **Output Layer:** A single neuron with a Sigmoid activation function, outputting a probability score between 0 and 1, indicating the likelihood of the text being hate speech.

### 4.3. Training & Evaluation

* **Optimizer:** (e.g., Adam optimizer with a learning rate scheduler).
* **Loss Function:** Binary Cross-Entropy (suitable for binary classification).
* **Metrics:** Given the likely class imbalance, primary evaluation metrics include:
    * **F1-score:** Harmonic mean of Precision and Recall.
    * **Recall:** Ability to correctly identify all hate speech instances (minimizing False Negatives).
    * **Precision:** Proportion of correctly identified hate speech instances among all instances predicted as hate speech (minimizing False Positives).
    * **Accuracy:** Overall correctness (less reliable with imbalanced data).
* **Early Stopping:** Used to prevent overfitting by monitoring validation loss.

---

## 5. Implementation Details & Results

### 5.1. Key Findings & Performance

<img width="1009" height="475" alt="image" src="https://github.com/user-attachments/assets/fa6f7d18-a6d2-4ee7-aaf6-19b94ad677fb" />



### 5.2. Limitations of the Current Model

* **Reliance on Word Embeddings:** The model's performance is heavily dependent on the quality and coverage of the word embeddings.
* **Contextual Limitations:** While BiLSTMs handle sequence context, Transformers are known to capture broader, more global contextual relationships, which could be beneficial for highly nuanced hate speech.
* **Computational Cost:** Training can be time-consuming for large datasets compared to simpler models.

---

## 6. Future Work (Hybrid BiLSTM-Transformer)

This BiLSTM project lays the groundwork for the next phase of my academic research:

1.  **Hybrid Architecture:** Integrating Transformer encoder blocks with BiLSTM layers to combine the strengths of both architectures. This aims to leverage the attention mechanisms of Transformers for global context while retaining the sequential processing power of BiLSTMs.
2.  **Pre-trained Language Models:** Experimenting with fine-tuning large pre-trained Portuguese language models (e.g., BERTimbau, mT5) as the embedding layer or as part of the overall architecture.
3.  **Multitask Learning:** Exploring multitask learning approaches where the model is simultaneously trained on related NLP tasks (e.g., sentiment analysis, offensive language detection) to improve generalization.
4.  **Adversarial Examples & Robustness:** Investigating the model's robustness against adversarial attacks and developing techniques to make the detection more resilient.

---

## Contact
Gabrielly Gomes - [Linkedin](https://www.linkedin.com/in/gabrielly-gomes-ml/) - gabrielly.gomes@ufpi.edu.br
