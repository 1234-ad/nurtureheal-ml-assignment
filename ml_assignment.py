#!/usr/bin/env python3
"""
NurtureHeal ML Assignment - Complete Machine Learning Pipeline
Dataset: Titanic Survival Dataset
Author: Your Name
Date: September 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== NURTUREHEAL ML ASSIGNMENT ===\n")
    
    # PART 1: Data Understanding & Preprocessing
    print("PART 1: DATA UNDERSTANDING & PREPROCESSING")
    print("=" * 50)
    
    # Load Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    
    # Data Cleaning
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)
    
    # Drop unnecessary columns
    df_clean = df_clean.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Feature Engineering
    df_clean['Family_Size'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_age_group = LabelEncoder()
    
    df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
    df_clean['Embarked'] = le_embarked.fit_transform(df_clean['Embarked'])
    df_clean['Age_Group'] = le_age_group.fit_transform(df_clean['Age_Group'])
    
    print(f"Cleaned Dataset Shape: {df_clean.shape}")
    print("Data preprocessing completed successfully!\n")
    
    # PART 2: Exploratory Data Analysis
    print("PART 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    print("Basic Statistics:")
    print(df_clean.describe())
    
    # Survival insights
    survival_rate = df_clean['Survived'].mean()
    print(f"\nOverall Survival Rate: {survival_rate:.2%}")
    
    # Create visualizations
    plt.figure(figsize=(15, 12))
    
    # 1. Survival Distribution
    plt.subplot(2, 3, 1)
    df_clean['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
    plt.title('Survival Distribution')
    plt.xlabel('Survived (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # 2. Age Distribution
    plt.subplot(2, 3, 2)
    plt.hist(df_clean['Age'], bins=30, alpha=0.7, color='blue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # 3. Correlation Heatmap
    plt.subplot(2, 3, 3)
    correlation_matrix = df_clean.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    
    # 4. Survival by Gender
    plt.subplot(2, 3, 4)
    gender_survival = df_clean.groupby('Sex')['Survived'].mean()
    gender_survival.index = ['Female', 'Male']
    gender_survival.plot(kind='bar', color=['pink', 'lightblue'])
    plt.title('Survival Rate by Gender')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    
    # 5. Survival by Class
    plt.subplot(2, 3, 5)
    class_survival = df_clean.groupby('Pclass')['Survived'].mean()
    class_survival.plot(kind='bar', color=['gold', 'silver', 'brown'])
    plt.title('Survival Rate by Class')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    
    # 6. Fare Distribution by Survival
    plt.subplot(2, 3, 6)
    survived = df_clean[df_clean['Survived'] == 1]['Fare']
    not_survived = df_clean[df_clean['Survived'] == 0]['Fare']
    plt.hist([not_survived, survived], bins=30, alpha=0.7, label=['Not Survived', 'Survived'], color=['red', 'green'])
    plt.title('Fare Distribution by Survival')
    plt.xlabel('Fare')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("EDA completed with 6 visualizations generated!\n")
    
    # PART 3: Model Building
    print("PART 3: MODEL BUILDING")
    print("=" * 50)
    
    # Prepare features and target
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for SVM and Logistic Regression
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:\n{cm}")
    
    # Create results comparison
    results_df = pd.DataFrame(results).T
    print(f"\nModel Comparison:")
    print(results_df.round(4))
    
    # PART 4: Hyperparameter Optimization
    print("\nPART 4: HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)
    
    # Optimize Random Forest (best performing model)
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Optimizing Random Forest...")
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    # Best model performance
    best_rf = rf_grid.best_estimator_
    y_pred_optimized = best_rf.predict(X_test)
    
    print(f"Best Parameters: {rf_grid.best_params_}")
    print(f"Best CV Score: {rf_grid.best_score_:.4f}")
    print(f"Optimized Test Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
    
    # Compare before and after optimization
    original_rf = RandomForestClassifier(random_state=42)
    original_rf.fit(X_train, y_train)
    original_pred = original_rf.predict(X_test)
    
    print(f"\nPerformance Comparison:")
    print(f"Original RF Accuracy: {accuracy_score(y_test, original_pred):.4f}")
    print(f"Optimized RF Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
    print(f"Improvement: {accuracy_score(y_test, y_pred_optimized) - accuracy_score(y_test, original_pred):.4f}")
    
    print("\n=== ASSIGNMENT COMPLETED SUCCESSFULLY ===")
    
    return df_clean, results_df, best_rf

if __name__ == "__main__":
    df_clean, results, best_model = main()