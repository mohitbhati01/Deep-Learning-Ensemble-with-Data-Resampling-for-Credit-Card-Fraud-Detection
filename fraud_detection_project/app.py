from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Base path for models
MODEL_PATH = 'models/'

# Load Models
try:
    xgb_model = joblib.load(os.path.join(MODEL_PATH, 'xgb_model.joblib'))
    rf_model = joblib.load(os.path.join(MODEL_PATH, 'rf_model.joblib'))
    lgb_model = joblib.load(os.path.join(MODEL_PATH, 'lgb_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    print("[SUCCESS] All Scikit-Learn (and XGB/LGB) models and scaler loaded successfully!")
except FileNotFoundError:
    print("[WARNING] Model files not found. Please run 'python src/train.py' first to generate them.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid or absent JSON body. Ensure "features" key is present.'}), 400
            
        print(f"DEBUG: Received features: {data['features']}")
        features = np.array(data['features']).reshape(1, -1)
        
        # Preprocessing (Scaling)
        features_scaled = scaler.transform(features)
        
        # Predictions (Probabilities for the 'Fraud' class)
        pred_xgb = xgb_model.predict_proba(features_scaled)[0][1]
        pred_rf = rf_model.predict_proba(features_scaled)[0][1]
        pred_lgb = lgb_model.predict_proba(features_scaled)[0][1]
        
        # Ensemble Average (Soft Voting)
        ensemble_prob = float((pred_xgb + pred_rf + pred_lgb) / 3.0)
        prediction = int(ensemble_prob > 0.5)
        
        print(f"DEBUG: Ensemble Prob: {ensemble_prob}, Prediction: {prediction}")
        
        return jsonify({
            'fraud_probability': round(ensemble_prob, 4),
            'prediction': 'Fraud' if prediction == 1 else 'Normal',
            'model_details': {
                'xgb_probability': round(float(pred_xgb), 4),
                'rf_probability': round(float(pred_rf), 4),
                'lgb_probability': round(float(pred_lgb), 4)
            }
        })

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR: {error_msg}")
        return jsonify({'error': str(e), 'traceback': error_msg}), 500
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Fraud Detection API is running'})

if __name__ == '__main__':
    # Ensure we are in the correct directory if running via script
    app.run(debug=True, port=5000)
