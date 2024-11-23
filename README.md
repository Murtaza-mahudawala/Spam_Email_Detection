
# Spam Email Detection

This project focuses on detecting spam emails using machine learning techniques. Multiple classification models were implemented and evaluated based on accuracy and precision metrics. The final model leverages the `Multinomial Naive Bayes` algorithm combined with a `TfidfVectorizer` for feature extraction. Additionally, advanced techniques like stacking and voting classifiers were used to further enhance performance.

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
9. [Acknowledgments](#Acknowledgements)
10. [Future Improvements](#future-work)
---

## Features

1. **Preprocessing:**
   - Tokenization
   - Stopword removal
   - Stemming using PorterStemmer

2. **EDA (Exploratory Data Analysis):**
   - Analyzed dataset characteristics (number of characters, words, sentences).
   - Visualized spam and ham (non-spam) word distributions using word clouds and histograms.

3. **Model Evaluation:**
   - Accuracy and precision metrics for multiple classifiers.
   - Comparative analysis of various machine learning models.

4. **Advanced Methods:**
   - Voting Classifier: Combines predictions of multiple models using a soft voting mechanism.
   - Stacking Classifier: Uses predictions of base models as input features for a final estimator.

---

## Dataset
The dataset contains labeled emails classified as:
- `0`: Non-spam (Ham)
- `1`: Spam  

### Statistics:
- Total emails: **[5572]**
- Ham emails: **[4848]**
- Spam emails: **[724]**

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

| Algorithm  | Accuracy | Precision |
|------------|----------|-----------|
| SVC        | 97.58%   | 97.48%    |
| K-Neighbors| 90.52%   | 100.00%   |
| Naive Bayes| 97.10%   | 100.00%   |
| Decision Tree | 93.23%| 83.33%    |
| Logistic Regression | 95.84% | 97.03% |
| Random Forest | 97.58% | 98.29%    |
| AdaBoost   | 96.03%   | 92.92%    |
| Bagging Classifier | 95.84% | 86.82% |
| Extra Trees| 97.49%   | 97.46%    |
| GBDT       | 94.68%   | 91.91%    |
| XGBoost    | 96.71%   | 92.62%    |

### Advanced Techniques

| Method         | Accuracy | Precision |
|----------------|----------|-----------|
| Voting Classifier | 98.16% | 99.17%    |
| Stacking Classifier | 97.78% | 93.23%    |

---

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
- Achieved an accuracy of **[97.7%]** on the test set.
- Precision for spam detection: **[96.3%]**.

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

1. Clone the repository:
   ```bash
   git clone https://github.com/Murtaza-mahudawala/Spam_Email_Detection.git
   cd Spam_Email_Detection
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application or Jupyter Notebook for exploration.

---

## Usage

1. Train the models using the provided script or notebook.
2. Test with new email texts to classify them as spam or ham.
3. Fine-tune parameters or explore advanced classifiers as needed.

---

## Results and Insights

The Voting Classifier achieved the highest precision (99.17%) with excellent accuracy (98.16%). It combines the strengths of SVC, Naive Bayes, and Extra Trees to provide robust performance.

---

## Acknowledgements

- Dataset sourced from open email datasets for spam detection.
- Libraries used: `scikit-learn`, `nltk`, `seaborn`, `matplotlib`, `xgboost`.

---

## Future Work

- Implement deep learning approaches like RNNs for email text classification.
- Enhance feature extraction with BERT or other transformer models.
