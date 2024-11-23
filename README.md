
# Spam Email Detection

## Overview
This project focuses on detecting spam emails using machine learning techniques. Various preprocessing steps, exploratory data analysis (EDA), and machine learning models were used to classify emails as spam or non-spam. The final model utilizes a VotingClassifier for optimal performance by combining the strengths of multiple algorithms.

---

## Table of Contents
1. [Features](#features)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Models and Evaluation](#models-and-evaluation)
5. [Final Model](#final-model)
6. [Results](#results)
7. [Setup and Requirements](#setup-and-requirements)
8. [Usage](#usage)
9. [Acknowledgments](#acknowledgments)

---

## Features
- **Text Preprocessing**: Tokenization, stemming, stop-word removal, and vectorization (TF-IDF and Bag of Words).
- **EDA**: Visualizations of word distributions and patterns in spam vs. non-spam emails.
- **Model Comparisons**: Implemented and evaluated various algorithms including Naive Bayes, SVM, Random Forest, and more.
- **Ensemble Learning**: Used VotingClassifier for final model selection.

---

## Dataset
The dataset contains labeled emails classified as:
- `0`: Non-spam (Ham)
- `1`: Spam  

### Statistics:
- Total emails: **[Include count from your dataset]**
- Ham emails: **[Include count]**
- Spam emails: **[Include count]**

---

## Preprocessing
1. **Data Cleaning**:
   - Removed punctuation, special characters, and converted text to lowercase.
2. **Tokenization**:
   - Split text into individual words.
3. **Feature Engineering**:
   - Extracted word frequencies and sentence counts.
   - Applied stemming to normalize word forms.
   - Vectorized text using TF-IDF with max features = 3000.

---

## Models and Evaluation
### Models Tested:
1. **Naive Bayes (Multinomial, Gaussian, Bernoulli)**  
   - Accuracy: **[Include score]**
   - Precision: **[Include score]**

2. **Support Vector Machine (SVM)**  
   - Accuracy: **[Include score]**
   - Precision: **[Include score]**

3. **Logistic Regression**  
   - Accuracy: **[Include score]**
   - Precision: **[Include score]**

4. **Decision Tree**  
   - Accuracy: **[Include score]**
   - Precision: **[Include score]**

5. **Random Forest**  
   - Accuracy: **[Include score]**
   - Precision: **[Include score]**

6. **VotingClassifier** (Final Model)  
   - Accuracy: **[Include score]**
   - Precision: **[Include score]**

### Performance Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Final Model
**VotingClassifier (Soft Voting)**  
The final model combines predictions from SVM, Naive Bayes, and Random Forest classifiers to achieve the best performance.  

---

## Results
- Achieved an accuracy of **[Include final accuracy]** on the test set.
- Precision for spam detection: **[Include precision]**.

---

## Setup and Requirements
### Prerequisites
- Python 3.8+
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - sklearn
  - nltk
  - wordcloud
  - xgboost

### Installation
1. Clone the repository:
   ```bash
   git clone [repository URL]
   cd [repository folder]
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. **Train the Model**:
   - Run the notebook to preprocess data and train models.
2. **Predict Spam Emails**:
   - Use the `model.pkl` and `vectorizer.pkl` files for inference.
   - Example code for inference:
     ```python
     import pickle
     model = pickle.load(open('model.pkl', 'rb'))
     vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
     email = "Congratulations! You've won a $1000 gift card. Claim now!"
     transformed_email = vectorizer.transform([email])
     prediction = model.predict(transformed_email)
     print("Spam" if prediction == 1 else "Ham")
     ```

---

## Acknowledgments
- Dataset sourced from [Mention source if available].
- Libraries: Scikit-learn, NLTK, Matplotlib, Seaborn.
