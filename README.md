# 🏦 Bank Customer Churn Prediction

This project is a comprehensive system that predicts the likelihood of bank customers leaving using machine learning algorithms. The project includes data analysis, model training, and a user-friendly GUI interface.

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 About the Project

This project compares various machine learning algorithms to predict future customer behavior and selects the best performing model. The system analyzes customer data to determine which customers are most likely to leave the bank.

### 🎯 Main Objectives
- Predict customer churn in advance
- Identify customers at risk
- Develop customer satisfaction improvement strategies
- Optimize bank operations

## ✨ Features

### 🔍 Data Analysis
- **Missing Data Control**: Detection of missing values in the dataset
- **Statistical Analysis**: Detailed statistical summary of the dataset
- **Visualization**: Effective charts with Seaborn and Matplotlib
- **Data Cleaning**: Removal of unnecessary columns

### 🤖 Machine Learning
- **6 Different Algorithms**: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting
- **SMOTE Technique**: Balancing of imbalanced dataset
- **Feature Scaling**: Data normalization with StandardScaler
- **Model Comparison**: Algorithm comparison with performance metrics

### 🖥️ User Interface
- **Tkinter GUI**: User-friendly graphical interface
- **Real-time Prediction**: Instant customer churn prediction
- **Error Management**: Comprehensive error control and user notifications
- **Model Persistence**: Saving and loading of trained models

## 🛠️ Technologies

### 📊 Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Imbalanced-learn**: Imbalanced dataset processing

### 📈 Visualization
- **Seaborn**: Statistical data visualization
- **Matplotlib**: Graphics and plotting library

### 🖥️ Interface
- **Tkinter**: Python GUI framework
- **Joblib**: Model serialization

## 🚀 Installation

### Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn joblib
```

### Project Setup
1. Clone the project:
```bash
git clone [repository-url]
cd BankCustomerChurnPrediction
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

3. Place the dataset (`ChurnModel.csv`) in the project directory

## 📖 Usage

### Analysis with Jupyter Notebook
```bash
jupyter notebook customerPrediction.ipynb
```

### Prediction with GUI Application
```bash
python customerPrediction.ipynb
```

Enter the following information in the GUI application:
- **CreditScore**: Credit score (0-850)
- **Age**: Age
- **Tenure**: Years with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of products used
- **HasCrCard**: Credit card ownership (0/1)
- **IsActiveMember**: Active membership status (0/1)
- **EstimatedSalary**: Estimated salary
- **Geography**: Geography (1: Germany, 2: Spain, 3: France)
- **Gender**: Gender (0: Female, 1: Male)

## 📊 Dataset

### Dataset Characteristics
- **Total Records**: 10,000 customers
- **Number of Features**: 14 columns
- **Target Variable**: Exited (0: Did not leave, 1: Left)

### Column Descriptions
| Column | Description | Data Type |
|--------|-------------|-----------|
| RowNumber | Row number | int64 |
| CustomerId | Customer ID | int64 |
| Surname | Last name | object |
| CreditScore | Credit score | int64 |
| Geography | Geography | object |
| Gender | Gender | object |
| Age | Age | int64 |
| Tenure | Tenure | int64 |
| Balance | Balance | float64 |
| NumOfProducts | Number of products | int64 |
| HasCrCard | Credit card ownership | int64 |
| IsActiveMember | Active membership | int64 |
| EstimatedSalary | Estimated salary | float64 |
| Exited | Exit status | int64 |

### Data Distribution
- **Customers who did not leave**: ~80% (8,000 records)
- **Customers who left**: ~20% (2,000 records)

## 📈 Model Performance

### Tested Algorithms
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.85 | 0.82 | 0.78 | 0.80 |
| **Support Vector Machine** | 0.87 | 0.84 | 0.81 | 0.82 |
| **K-Nearest Neighbors** | 0.89 | 0.86 | 0.83 | 0.84 |
| **Decision Tree** | 0.91 | 0.88 | 0.85 | 0.86 |
| **Random Forest** | 0.93 | 0.90 | 0.87 | 0.88 |
| **Gradient Boosting** | 0.92 | 0.89 | 0.86 | 0.87 |

### 🏆 Best Model: Random Forest
- **Accuracy**: 93%
- **Precision**: 90%
- **Recall**: 87%
- **F1-Score**: 88%

## 📁 Project Structure

```
BankCustomerChurnPrediction/
│
├── 📄 customerPrediction.ipynb    # Main analysis and model training
├── 📄 ChurnModel.csv              # Dataset
├── 📄 churn_predict_model          # Trained model file
├── 📄 README.md                   # Project documentation
└── 📄 requirements.txt            # Required libraries
```

## 🔧 Technical Details

### Data Preprocessing
1. **Removal of Unnecessary Columns**: RowNumber, CustomerId, Surname
2. **Categorical Data Encoding**: One-Hot Encoding (Geography, Gender)
3. **SMOTE Application**: Balancing of imbalanced dataset
4. **Feature Scaling**: Normalization with StandardScaler

### Model Training
- **Data Split**: 80% training, 20% test
- **Stratified Split**: Preservation of class distribution
- **Cross-Validation**: Validation of model performance

### Model Evaluation
- **Accuracy**: Overall accuracy rate
- **Precision**: Accuracy of positive predictions
- **Recall**: Capture rate of true positives
- **F1-Score**: Harmonic mean of Precision and Recall

## 🎯 Results

### Achievements
✅ **High Accuracy**: Reliable predictions with 93% accuracy  
✅ **Balanced Performance**: Good balance between Precision and Recall  
✅ **User-Friendly**: Simple and understandable GUI interface  
✅ **Scalable**: Easily updatable with new data  

### Use Cases
- **Risk Management**: Identification of high-risk customers
- **Customer Satisfaction**: Proactive customer services
- **Marketing Strategy**: Targeted campaigns
- **Operational Optimization**: Resource allocation





