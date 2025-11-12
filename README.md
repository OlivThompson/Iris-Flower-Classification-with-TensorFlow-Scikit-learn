
# Iris Classification with SVM and TensorFlow

This project is an **adaptation of a previous machine learning assignment**, expanded to include a **TensorFlow-based neural network model** in addition to the original **Scikit-learn SVM** implementation. The project demonstrates data preprocessing, visualization, training, and evaluation on the classic **Iris flower dataset**.

***

### Overview

The purpose of this project is to explore two different approaches to classifying the Iris dataset:

- A **traditional SVM (Support Vector Machine)** classifier using Scikit-learn  
- A **feed-forward neural network** implemented with TensorFlow’s Keras API  

The project allows comparison between conventional machine learning methods and simple deep learning models on the same dataset.

***

### Features

- Loads and visualizes the Iris dataset using **Pandas** and **Matplotlib**
- Standardizes numeric features using **StandardScaler**
- Implements and evaluates an **SVM classifier** with accuracy and confusion matrix reports
- Builds and trains a **neural network model** in TensorFlow for classification
- Compares model performance between classical and deep learning methods

***

### Technologies Used

- **Python 3.x**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **TensorFlow / Keras**

***

### Data

The **Iris dataset** is a well-known benchmark dataset in machine learning. It contains measurements of 150 iris flowers from three species:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

Each sample includes four features:
- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

***

### How to Run

1. **Install dependencies:**
   ```bash
   pip install pandas matplotlib scikit-learn tensorflow
   ```

2. **Run the script:**
   ```bash
   python iris_classification_tf.py
   ```

3. **Expected outputs:**
   - A scatter matrix visualization of the dataset
   - SVM model accuracy and classification report
   - Neural network training output and accuracy evaluation

***

### Results

The assignment extension demonstrates that:
- The **SVM model** achieves reliable baseline accuracy using Scikit-learn code.  
- The **neural network** implemented in TensorFlow can achieve similar or improved performance through multiple training epochs.  

