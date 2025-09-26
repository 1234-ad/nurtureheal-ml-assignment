#!/usr/bin/env python3
"""
NurtureHeal ML Assignment - Flask Deployment App
Titanic Survival Prediction Web Application
"""

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load and prepare the model (this would normally be loaded from a saved model)
def prepare_model():
    # Load Titanic dataset for training
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Data preprocessing (same as in main script)
    df_clean = df.copy()
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)
    df_clean = df_clean.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df_clean['Family_Size'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_age_group = LabelEncoder()
    
    df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
    df_clean['Embarked'] = le_embarked.fit_transform(df_clean['Embarked'])
    df_clean['Age_Group'] = le_age_group.fit_transform(df_clean['Age_Group'])
    
    # Prepare features and target
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    
    # Train optimized Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    
    return model, le_sex, le_embarked, le_age_group

# Initialize model and encoders
model, le_sex, le_embarked, le_age_group = prepare_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        has_cabin = int(request.form['has_cabin'])
        
        # Feature engineering
        family_size = sibsp + parch + 1
        
        # Age group encoding
        if age <= 18:
            age_group = 0  # Child
        elif age <= 35:
            age_group = 3  # Young Adult
        elif age <= 60:
            age_group = 1  # Adult
        else:
            age_group = 2  # Senior
        
        # Encode categorical variables
        sex_encoded = 1 if sex == 'male' else 0
        embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]
        
        # Create feature array
        features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, 
                            embarked_encoded, has_cabin, family_size, age_group]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        result = {
            'prediction': 'Survived' if prediction == 1 else 'Did not survive',
            'probability': f"{probability[1]:.2%}" if prediction == 1 else f"{probability[0]:.2%}",
            'confidence': f"{max(probability):.2%}"
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        
        # Extract features
        pclass = data['pclass']
        sex_encoded = 1 if data['sex'] == 'male' else 0
        age = data['age']
        sibsp = data['sibsp']
        parch = data['parch']
        fare = data['fare']
        embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[data['embarked']]
        has_cabin = data['has_cabin']
        family_size = sibsp + parch + 1
        
        # Age group
        if age <= 18:
            age_group = 0
        elif age <= 35:
            age_group = 3
        elif age <= 60:
            age_group = 1
        else:
            age_group = 2
        
        features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, 
                            embarked_encoded, has_cabin, family_size, age_group]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'survival_probability': float(probability[1]),
            'confidence': float(max(probability))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)