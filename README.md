# Fall Detection using SisFall Database

- [Fall Detection using SisFall Database](#fall-detection-using-sisfall-database)
  - [Introduction](#introduction)
  - [Key Ideas](#key-ideas)
  - [Technologies and Best Practices](#technologies-and-best-practices)
  - [Installation](#installation)
  - [Data](#data)
  - [Usage](#usage)
  - [Results](#results)
  - [Conclusion](#conclusion)

## Introduction

Welcome to the Fall Detection project using the SisFall database. This Python-based project aims to detect falls from sensor data. By leveraging machine learning algorithms such as Neural Networks, Logistic Regression, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN), it provides a practical solution for fall detection with applications in healthcare, eldercare, and safety monitoring.

## Key Ideas

The project's efficiency and accuracy are driven by several key ideas:

1. **Working with Addresses**: The model optimizes performance by working with file addresses instead of the entire dataset whenever possible. This approach significantly improves processing speed.

2. **Balancing the Dataset**: Recognizing that the number of fall records is much lower than non-fall records, the project balances the dataset to ensure fair and accurate model training.

3. **Ensemble Concept**: An ensemble concept is applied, combining predictions from KNN and Neural Network models. This fusion of models enhances the accuracy of fall detection.

## Technologies and Best Practices

- **Python**: The project is coded in Python, a versatile language for machine learning and data analysis.

- **Machine Learning Libraries**: It uses libraries like Scikit-learn and TensorFlow for model creation and training.

- **Feature Extraction**: Relevant features are extracted from sensor data, improving the model's accuracy.

- **Data Pre-processing**: The dataset undergoes pre-processing, normalization, and balancing for enhanced model performance.

## Installation

Before running the code, make sure to install the required libraries using pip:

```bash
pip install scikit-learn tensorFlow pandas numpy
```

## Data
The code downloads and utilizes two datasets: SisFall_dataset and SisFall_enhanced. These datasets are used for training and testing the fall detection models.

## Usage
The project follows a structured workflow:

1. **Data Pre-processing**: The code focuses on data pre-processing, reading and extracting features from sensor data. The dataset is then split into training and testing sets.

2. **Model Training**: Multiple machine learning models (Neural Networks, Logistic Regression, SVM, and KNN) are trained and evaluated.

3. **Balancing Data**: Balancing the dataset is critical to enhancing model performance. The code achieves this for each model.

4. **Ensemble Concept**: Predictions from KNN and Neural Network models are combined using an ensemble concept to enhance fall detection accuracy.

## Results
The project provides a comprehensive analysis of each model's performance, including precision, recall, and F1-score. Notably, after balancing the dataset, the ensemble concept and KNN model yielded the best results.

## Conclusion
This fall detection project holds great promise for applications in healthcare, safety monitoring, and more. By combining the power of machine learning, feature extraction, and ensemble modeling, it offers an efficient and accurate solution for fall detection. Feel free to explore the code and results, and your feedback and contributions are highly encouraged.


![alt text](https://github.com/MmdFt/Fall-Detection-System/blob/master/Models'%20preformance%20summary.png?raw=true)

