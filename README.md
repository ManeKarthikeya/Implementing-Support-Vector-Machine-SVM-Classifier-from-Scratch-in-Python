# 🤖 Implementing Support Vector Machine (SVM) Classifier from Scratch in Python

## 🧮 Complete Implementation of SVM Algorithm with Gradient Descent

A **from-scratch implementation** of the Support Vector Machine (SVM) classifier algorithm using pure Python and NumPy. This educational project demonstrates the fundamental mathematics behind one of the most powerful classification algorithms, complete with gradient descent optimization and practical diabetes prediction application.

## 🎯 Project Overview

This project implements an SVM classifier **without relying on machine learning libraries** like scikit-learn. It provides a deep understanding of:
- The mathematical foundations of SVM classification
- Gradient Descent optimization for SVM
- Hinge loss and margin maximization concepts
- Parameter updates for weights and bias
- Model training and prediction mechanics

## 📚 Mathematical Foundations

### 📐 SVM Hyperplane Equation
```
y = wx - b
```
Where:
- **w**: Weight vector (determines orientation of hyperplane)
- **b**: Bias term (determines offset from origin)
- **x**: Input feature vector
- **y**: Predicted class label

### ⚙️ Gradient Descent for SVM

Gradient Descent is an optimization algorithm used for minimizing the loss function by iteratively updating parameters:

**Parameter Update Rules:**
```
w = w - α * dw
b = b - α * db
```

**Learning Rate (α)**: Tuning parameter that determines step size at each iteration while moving toward minimum loss.

### 🔍 Hinge Loss & Gradient Computation

The SVM optimization uses hinge loss with regularization:

**Condition Check:**
```
condition = y_label[i] * (np.dot(x_i, w) - b) >= 1
```

**Gradient Calculation:**
- **If condition met** (correctly classified with margin):
  ```
  dw = 2 * λ * w
  db = 0
  ```

- **If condition violated** (misclassified or within margin):
  ```
  dw = 2 * λ * w - np.dot(x_i, y_label[i])
  db = y_label[i]
  ```

Where:
- **λ (lambda_parameter)**: Regularization parameter to control overfitting
- **y_label**: Encoded labels (-1, 1) for SVM formulation

## 🛠️ Technical Implementation

### 🏗️ Custom SVM Classifier Class

#### Class Structure:
```python
class SVM_classifier():
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        # Initialize hyperparameters
    
    def fit(self, X, Y):
        # Training function with gradient descent
    
    def update_weights(self):
        # Gradient descent weight updates with hinge loss
    
    def predict(self, X):
        # Make predictions using learned parameters
```

#### Core Methods:
1. **`__init__`**: Initialize learning rate, iterations, and regularization parameter
2. **`fit`**: Train model using gradient descent with hinge loss
3. **`update_weights`**: Compute gradients based on margin violations and update parameters
4. **`predict`**: Generate predictions using learned hyperplane

### 🔄 Gradient Descent Implementation
```python
def update_weights(self):
    # Label encoding (convert 0/1 to -1/1 for SVM)
    y_label = np.where(self.Y <= 0, -1, 1)
    
    for index, x_i in enumerate(self.X):
        # Check margin condition
        condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
        
        if condition == True:
            # Correctly classified with sufficient margin
            dw = 2 * self.lambda_parameter * self.w
            db = 0
        else:
            # Margin violation or misclassification
            dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
            db = y_label[index]
        
        # Update parameters
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
```

## 📊 Dataset & Application

### 🏥 Diabetes Prediction Application
**Dataset**: PIMA Indians Diabetes Database
- **Features**: 8 medical predictor variables
- **Target**: Diabetes outcome (0 = Non-diabetic, 1 = Diabetic)
- **Samples**: 768 patient records

**Feature Set:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### 🧪 Model Training Results
```
Hyperparameters Used:
- Learning Rate: 0.001
- Number of Iterations: 1000
- Lambda Parameter (Regularization): 0.01
```

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install numpy pandas scikit-learn
```

### Installation & Usage

1. **Clone the repository**:
```bash
git clone https://github.com/ManeKarthikeya/Implementing-SVM-from-Scratch.git
cd Implementing-SVM-from-Scratch
```

2. **Run the diabetes prediction system**:
```bash
python implementing_support_vector_machine_classifier_from_scratch_in_python.py
```

3. **Input patient data** when prompted:
```python
# Example input format:
# (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Output:
# [1] - The Person is diabetic (or)
# [0] - The person is not diabetic
```

### Manual Usage Example
```python
# Import custom SVM class
from svm_classifier import SVM_classifier

# Create and train model
classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
classifier.fit(X_train, Y_train)

# Make predictions
predictions = classifier.predict(X_test)
```

## 📁 Project Structure

```
SVM-from-Scratch/
├── implementing_support_vector_machine_classifier_from_scratch_in_python.py  # Main implementation
├── diabetes.csv                                     # PIMA Diabetes dataset
├── requirements.txt                                 # Python dependencies (add on you'r own)
└── README.md                                       # Project documentation
```

## 🧪 Model Evaluation

### 📊 Performance Metrics
- **Training Accuracy**: ~78-82% on training data
- **Test Accuracy**: ~75-80% on unseen test data

### 🔍 Accuracy Score Calculation
```python
# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
```

## 🎯 Educational Value

### 🎓 Learning Objectives
1. **Understanding SVM Mathematics**
   - Hyperplane separation concept
   - Margin maximization intuition
   - Hinge loss derivation
   - Regularization trade-offs

2. **Implementing Gradient Descent for SVM**
   - Parameter update mechanics
   - Learning rate impact
   - Margin violation handling
   - Convergence criteria

3. **Custom Algorithm Development**
   - Object-oriented programming for ML
   - Numerical computation with NumPy
   - Model evaluation techniques
   - Hyperparameter tuning

### 🔍 Key Insights
- **Margin Concept**: SVM aims to maximize the margin between classes
- **Support Vectors**: Only points near decision boundary influence the model
- **Regularization**: Lambda parameter controls bias-variance trade-off
- **Label Encoding**: SVM requires labels as -1 and 1, not 0 and 1
