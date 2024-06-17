# COVID Sentiment Analysis 

This repository contains a comprehensive Natural Language Processing (NLP) project focused on sentiment analysis for COVID-related data. The project involves the classification of sentiments into five distinct classes, offering a nuanced understanding of public opinion during the COVID-19 pandemic.

## Overview

The project begins with an in-depth exploration of the dataset, aiming to understand the structure, trends, and patterns within the data. This exploratory data analysis (EDA) phase is crucial for gaining insights into the sentiment distribution and identifying any potential issues such as class imbalance.

## Data Preprocessing

To address any imbalances in the dataset, several techniques are employed, including undersampling, oversampling, and text augmentation. These techniques ensure that the model is trained on a balanced representation of each sentiment class, thereby improving its ability to generalize.

## Models

### Machine Learning Models

The project employs a range of machine learning algorithms for multi-class classification. This includes basic algorithms such as logistic regression and decision trees, as well as more advanced boosting algorithms like XGBoost. The choice of algorithms allows for a comparative analysis of performance and robustness across different approaches.

### Deep Learning Models

In addition to traditional machine learning algorithms, the project explores deep learning techniques for sentiment analysis. Convolutional Neural Networks (CNNs) are utilized with custom embedding layers as well as pre-trained GloVe embeddings to capture semantic relationships within the text. Furthermore, transformer-based models are employed to leverage the power of self-attention mechanisms for better understanding of contextual information.

## Evaluation Metrics

In this project, the following metrics are considered for model evaluation. These metrics are chosen since there isn't a clear business understanding of which class is more important or where misidentification of a class is more costly:

1. **Accuracy**: Provides a quick view of the model's effectiveness across all classes.

2. **Precision**:
   - **Per Class Precision**: Important for understanding how well the model predicts each class without labeling too many negatives as positives. Crucial for classes like 'Extremely Negative' where misclassifications can be more severe.
   - **Average Precision**: Provides an overall measure of precision across all classes.

3. **Recall**: Vital for ensuring that the model is capable of capturing the majority of positive samples in each class.

4. **Confusion Matrix**: Provides an intrinsic analysis of mislabeling, helping to understand which classes are being misclassified.

ROC AUC can be utilized if a one-vs-rest approach is taken, however, it's not implemented in this version.

F1-score is not used as the classes have been balanced, and precision and recall are calculated separately for better understanding.

## Conclusion

For future iterations of the project, the following improvements and recommendations could be considered:

- Tailor the model to the specific needs of the business scenario at hand, focusing on the **most critical classes** for accuracy improvement. This could involve collaborating closely with domain experts to understand which misclassifications are most costly or harmful.

- Reduce the number of classes by combining "Positive" with "Extremely Positive" and "Negative" with "Extremely Negative". This simplification could improve the model's performance by reducing the complexity of the classification task. To address class imbalance, augment data for underrepresented classes that would be neutral in this case. 

- Leverage pre-trained models such as https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest from Hugging Face's model repository. These models, already fine-tuned on similar data, can provide a strong starting point and may yield better performance out-of-the-box compared to training a model from scratch.

- Conduct a thorough error analysis to understand the types of misclassifications occurring and refine the preprocessing, feature engineering, or model architecture accordingly.

