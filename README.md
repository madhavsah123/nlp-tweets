# 🚨 NLP with Disaster Tweets

## 🌟 Project Overview

During emergencies like earthquakes, floods, or wildfires, millions of tweets appear within minutes. Only a fraction report actual incidents—many are jokes, opinions, or unrelated chatter. Quickly identifying real disaster tweets is critical for first responders and crisis management.

---

## 🎯 Purpose

The goal of this project is to build a machine learning model that can read a tweet and instantly predict whether it describes a real disaster (`1`) 🌍🔥 or not a disaster (`0`) 🙅‍♂️.

---

## 🤖 How It Helps

Once trained, the model can scan massive, real-time Twitter streams and automatically highlight the tweets that truly matter. This enables rapid situational awareness and helps first responders react faster when every second counts.

---

## 🛠️ Libraries & Tools Used

- **Python 3**
- **Pandas**, **NumPy** for data handling
- **NLTK** for text preprocessing and sentiment analysis
- **Scikit-learn** for machine learning and vectorization
- **Keras** and **TensorFlow** for deep learning models
- **Gensim** for Word2Vec embeddings
- **Transformers** (HuggingFace) for BERT
- **Matplotlib**, **Seaborn** for visualization
- **WordCloud** for word cloud generation

---

## 📊 Data Analysis & Visualization

- **Class Distribution**: Visualized the count of real vs. fake disaster tweets.
- **Tweet Length Analysis**: Histograms of character and word length distributions.
- **Stop Word & Punctuation Analysis**: Most common stopwords and punctuations in each class.
- **N-grams**: Top bigrams and trigrams visualized for insight into frequent word pairs/triplets.

---

## 🧹 Preprocessing

- Removal of URLs and non-essential characters using regex.
- Lowercasing, contraction handling, and spell correction.
- Removal of stopwords and short words.
- Lemmatization for normalization.

---

## ☁️ Word Clouds

- Separate word clouds for real disaster and non-disaster tweets to visualize the most frequent words in each category.

---

## 🔡 Feature Extraction

- **TF-IDF** and **CountVectorizer** used to convert text data into numerical features for model training.
- **SVD** used for dimensionality reduction.

---

## 🤖 Model Building

### Classical ML Models

- Random Forest 🌲
- Decision Tree 🌳
- Logistic Regression 📈
- Linear SVC 💻
- Ridge Classifier 🏔️
- SVC (Support Vector Classifier) 🧮

Performance compared using F1 score and accuracy. 

### Deep Learning

- **BERT** (Bidirectional Encoder Representations from Transformers) for advanced sequence classification.

---

## 📈 Results

- Compared classical machine learning models using F1 scores, training, and test accuracies.
- Visualized model performance via line plots.
- Implemented BERT for further improvements.

---

## 🚀 Takeaways

This project demonstrates an end-to-end NLP workflow for disaster tweet classification:
- Exploratory analysis 📊
- Data cleaning 🧹
- Feature engineering 🏗️
- Model training & comparison 🤖
- Visualization & interpretation 🎨

With such a model, organizations can turn social media chaos into actionable intelligence—making every second count when lives are on the line. 🕒💡

---

## 👨‍💻 Author

- Madhav Sah

---

## 📂 Source

- [Original Notebook on GitHub](https://github.com/madhavsah123/nlp-tweets/blob/ca784f10f5573cdd314be2a04e0c8dc042cb7b0b/twitter_disaster_nlp_by_madhav_sah-5.ipynb)

---
