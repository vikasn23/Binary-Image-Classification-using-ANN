# Image Classification for Raveling and Non-Raveling Images

This repository contains the code and steps for image classification of "Raveling" and "Non-Raveling" images using a deep learning model. The images are pre-processed to extract both color-based and GLCM (Gray-Level Co-occurrence Matrix) texture-based features. These features are then used to train and evaluate a neural network model for binary classification.

## Table of Contents

1. [Feature Extraction for Training Dataset](#feature-extraction-for-training-dataset)
2. [Feature Extraction for Test Dataset](#feature-extraction-for-test-dataset)
3. [Image Classification Model](#image-classification-model)
4. [Evaluation and Testing](#evaluation-and-testing)
5. [Final Predictions](#final-predictions)

---

## Feature Extraction for Training Dataset

The feature extraction process involves:

- **Color-based Features**: 18 features for each image channel (Red, Green, Blue) are extracted, including mean, standard deviation, skewness, kurtosis, entropy, and range.
- **GLCM Texture-based Features**: 16 features are derived from the Gray-Level Co-occurrence Matrix (GLCM) at 4 different angles (0°, 45°, 90°, and 135°).

### Steps:
1. **Load Images**: Images from the "Raveling" folder are read and processed.
2. **Extract Features**: Both color-based and texture-based features are extracted from each image.
3. **Store Features**: The extracted features are saved in a CSV file (`raveling_image_features.csv`).

---

## Feature Extraction for Test Dataset

The process for extracting features from the test dataset is similar to the training dataset. It extracts:

- **Color-based Features**
- **GLCM Texture-based Features**

### Steps:
1. **Load Images**: Unlabeled images from the "Raveling_non_raveling" folder are read.
2. **Extract Features**: Features are extracted from each image.
3. **Store Features**: The features are saved in a CSV file (`test_images_features_.csv`).

---

## Image Classification Model

The classification model is built using Keras and the extracted features. The steps are as follows:

### Data Preparation:
1. **Load the CSV**: The `images_for_training.csv` file contains the extracted features and target labels (Raveling/Non-Raveling).
2. **Data Preprocessing**:
    - Separate features and target labels.
    - Encode target labels (Raveling -> 1, Non-Raveling -> 0).
    - Split the dataset into training and testing sets (80% training, 20% testing).
    - Standardize the features using `StandardScaler` to improve training performance.

### Model Architecture:
- **Input Layer**: The input layer accepts 34 features (18 color features + 16 GLCM features).
- **Hidden Layers**: Three hidden layers with LeakyReLU activations, BatchNormalization, and Dropout for regularization.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification.
- **Loss Function**: `binary_crossentropy`.
- **Optimizer**: Adam optimizer for efficient training.
- **Metrics**: Accuracy is used to evaluate the model performance.

### Training:
The model is trained using the training data and the best weights are saved using the `ModelCheckpoint` callback.

---

## Evaluation and Testing

After training, the model is evaluated on the test set:

- **Classification Report**: Precision, recall, F1-score, and support for both classes (Raveling and Non-Raveling) are displayed.
- **Accuracy**: The overall accuracy of the model is calculated.

Example output:
```text
              precision    recall  f1-score   support
           0       0.94      0.97      0.95        61
           1       0.97      0.95      0.96        79

    accuracy                           0.96       140
   macro avg       0.96      0.96      0.96       140
weighted avg       0.96      0.96      0.96       140

Accuracy: 0.9571
