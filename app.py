from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and preprocessing
model = None
scaler = None
label_encoders = {}

class LoanDefaultPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_sample_data(self):
        """Create sample training data for demonstration"""
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'Client_Income': np.random.normal(50000, 20000, n_samples),
            'Car_Owned': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Bike_Owned': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Active_Loan': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'House_Own': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'Child_Count': np.random.poisson(1, n_samples),
            'Credit_Amount': np.random.normal(200000, 100000, n_samples),
            'Loan_Annuity': np.random.normal(15000, 5000, n_samples),
            'Client_Income_Type': np.random.choice(['Working', 'Commercial', 'Pension'], n_samples),
            'Client_Education': np.random.choice(['Secondary', 'Higher', 'Incomplete'], n_samples),
            'Client_Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'Age_Days': np.random.normal(45*365, 10*365, n_samples),
            'Employed_Days': np.random.normal(5*365, 3*365, n_samples),
            'Score_Source_1': np.random.normal(650, 100, n_samples),
            'Score_Source_2': np.random.normal(650, 100, n_samples),
            'Score_Source_3': np.random.normal(650, 100, n_samples),
            'Social_Circle_Default': np.random.poisson(1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable with realistic relationships
        default_prob = (
            df['Client_Income'] / 10000 * -0.1 +
            df['Credit_Amount'] / 100000 * 0.3 +
            df['Child_Count'] * 0.2 +
            (df['Age_Days'] < 30*365) * 0.5 +
            (df['Employed_Days'] < 365) * 0.8 +
            np.random.normal(0, 1, n_samples)
        )
        
        df['Default'] = (default_prob > default_prob.mean()).astype(int)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training/prediction"""
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_columns = ['Client_Income_Type', 'Client_Education', 'Client_Marital_Status']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Select important features
        important_features = [
            'Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 
            'Employed_Days', 'Score_Source_1', 'Score_Source_2', 'Score_Source_3',
            'Social_Circle_Default', 'Child_Count', 'Car_Owned', 'House_Own',
            'Client_Income_Type', 'Client_Education'
        ]
        
        # Keep only features that exist in dataframe
        available_features = [f for f in important_features if f in df_processed.columns]
        
        return df_processed[available_features]
    
    def train_model(self):
        """Train the Random Forest model"""
        print("Creating sample data...")
        df = self.create_sample_data()
        
        print("Preprocessing data...")
        X = self.preprocess_data(df)
        y = df['Default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        return self.model, self.scaler, self.label_encoders

def init_model():
    """Initialize or load the model"""
    global model, scaler, label_encoders
    
    # UPDATED: Use your actual filename
    model_path = 'Model/model.pkl'  # Changed to match your file
    preprocessing_path = 'Model/preprocessing.pkl'
    
    # Create model directory if it doesn't exist
    os.makedirs('Model', exist_ok=True)
    
    try:
        # Try to load existing model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessing_path, 'rb') as f:
            preprocessing_data = pickle.load(f)
            scaler = preprocessing_data['scaler']
            label_encoders = preprocessing_data['label_encoders']
        print("Model loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Training new model...")
        predictor = LoanDefaultPredictor()
        model, scaler, label_encoders = predictor.train_model()
        
        # Save model and preprocessing objects
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(preprocessing_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'label_encoders': label_encoders}, f)
        print("Model trained and saved successfully!")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        predictor = LoanDefaultPredictor()
        model, scaler, label_encoders = predictor.train_model()
        
        # Save model and preprocessing objects
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(preprocessing_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'label_encoders': label_encoders}, f)
        print("Model trained and saved successfully!")

# Rest of your routes remain the same...
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert to appropriate data types
        input_data = {}
        for key, value in form_data.items():
            if value == '':
                input_data[key] = 0
            elif key in ['Client_Income_Type', 'Client_Education', 'Client_Marital_Status']:
                input_data[key] = value
            else:
                try:
                    input_data[key] = float(value)
                except:
                    input_data[key] = value
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess input data
        predictor = LoanDefaultPredictor()
        predictor.scaler = scaler
        predictor.label_encoders = label_encoders
        
        processed_data = predictor.preprocess_data(input_df)
        
        # Ensure all required features are present
        expected_features = [
            'Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days',
            'Employed_Days', 'Score_Source_1', 'Score_Source_2', 'Score_Source_3',
            'Social_Circle_Default', 'Child_Count', 'Car_Owned', 'House_Own',
            'Client_Income_Type', 'Client_Education'
        ]
        
        for feature in expected_features:
            if feature not in processed_data.columns:
                processed_data[feature] = 0
        
        processed_data = processed_data[expected_features]
        
        # Scale features
        scaled_data = scaler.transform(processed_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Get feature importance
        feature_importance = dict(zip(expected_features, model.feature_importances_))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'prediction': 'High Risk of Default' if prediction == 1 else 'Low Risk of Default',
            'probability': round(probability * 100, 2),
            'risk_level': 'high' if prediction == 1 else 'low',
            'feature_importance': sorted_importance[:5]  # Top 5 features
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Initializing loan default prediction model...")
    init_model()
    app.run(debug=True, host='0.0.0.0', port=5001)