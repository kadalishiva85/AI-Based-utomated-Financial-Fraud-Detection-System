from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global Variables for models and scaler
xgb_model = None
logreg_model = None
scaler = None


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/train', methods=['POST'])
def train():
    try:
        if 'csvFile' not in request.files:
            return jsonify({'error': 'No CSV file uploaded'}), 400

        csv_file = request.files['csvFile']
        df = pd.read_csv(csv_file)

        if 'Class' not in df.columns:
            return jsonify({'error': "Dataset must contain a 'Class' column."}), 400

        # Preprocessing
        X = df.drop('Class', axis=1)
        y = df['Class']
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )

        # Scaling
        global scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Training models
        global xgb_model, logreg_model
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=50,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)

        logreg_model = LogisticRegression(random_state=42)
        logreg_model.fit(X_train_scaled, y_train)

        # Evaluation
        xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test_scaled))
        logreg_accuracy = accuracy_score(y_test, logreg_model.predict(X_test_scaled))

        xgb_report = classification_report(y_test, xgb_model.predict(X_test_scaled), output_dict=True)
        logreg_report = classification_report(y_test, logreg_model.predict(X_test_scaled), output_dict=True)

        return jsonify({
            'message': 'Models are successfully trained!',
            'xgb_accuracy': round(xgb_accuracy * 100, 2),
            'logreg_accuracy': round(logreg_accuracy * 100, 2),
            'xgb_report': xgb_report,
            'logreg_report': logreg_report
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not xgb_model or not logreg_model or not scaler:
            return jsonify({'error': 'Models are not trained yet. Please train them first.'}), 400

        if 'csvFile' not in request.files:
            return jsonify({'error': 'No CSV file uploaded'}), 400

        csv_file = request.files['csvFile']
        df = pd.read_csv(csv_file)

        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        df = df.fillna(df.mean())
        df_scaled = scaler.transform(df)

        logreg_pred = logreg_model.predict(df_scaled)

        legal_transactions = int((logreg_pred == 0).sum())
        fraud_transactions = int((logreg_pred == 1).sum())

        return jsonify({
            'legal_transactions': legal_transactions,
            'fraud_transactions': fraud_transactions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
