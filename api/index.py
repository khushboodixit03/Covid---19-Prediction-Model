from flask import Flask, request, jsonify, send_file
import os
import joblib
import base64
from datetime import datetime
from fpdf import FPDF
import io

app = Flask(__name__)

# Config
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_outputs')
MODEL_PATH = os.path.join(MODEL_DIR, "covid_model.joblib")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
FEATURES = ["Oxygen", "PulseRate", "Temperature"]

# Load models globally (cache for Vercel)
model = None
le = None

def load_models():
    global model, le
    if model is None or le is None:
        try:
            model = joblib.load(MODEL_PATH)
            le = joblib.load(LE_PATH)
        except Exception as e:
            print(f"Error loading models: {e}")
            raise e

def generate_pdf_bytes(vitals_dict, prediction_label, prob=None, photo_base64=None, patient_id=None, patient_name=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "COVID-19 Prediction Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Timestamp: {datetime.now().isoformat(sep=' ', timespec='seconds')}", ln=True)
    if patient_name:
        pdf.cell(0, 8, f"Patient Name: {patient_name}", ln=True)
    if patient_id:
        pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"PREDICTION: {prediction_label}", ln=True)
    if prob is not None:
        pdf.cell(0, 7, f"Confidence: {prob}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 7, "Provided Vitals:", ln=True)
    for k, v in vitals_dict.items():
        pdf.cell(0, 6, f" - {k}: {v}", ln=True)
    pdf.ln(4)

    if photo_base64:
        try:
            # Decode base64 image to temp file for FPDF
            import tempfile
            img_data = base64.b64decode(photo_base64.split(',')[1])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name
            
            pdf.image(tmp_path, x=10, y=pdf.get_y(), w=80)
            os.remove(tmp_path)
        except Exception as e:
            pdf.cell(0, 6, f"(Could not include photo: {e})", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 5, "Disclaimer: This is a prototype model output. This prediction should NOT be used as a medical diagnosis.")
    
    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        load_models()
        data = request.json
        
        # Extract Vitals
        vitals = {
            "Oxygen": float(data.get('oxygen')),
            "PulseRate": float(data.get('pulse')),
            "Temperature": float(data.get('temperature'))
        }
        
        # Prepare feature vector
        fv = [vitals[k] for k in FEATURES]
        
        # Predict
        pred_idx = model.predict([fv])[0]
        pred_label = le.classes_[int(pred_idx)]
        
        pred_prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([fv])[0]
            pred_prob = str(round(float(max(probs)), 4))

        # Generate PDF
        pdf_bytes = generate_pdf_bytes(
            vitals, 
            pred_label, 
            prob=pred_prob, 
            photo_base64=data.get('photo'), # expect data:image/jpeg;base64,...
            patient_id=data.get('patientId'),
            patient_name=data.get('patientName')
        )
        
        # Encode PDF to base64 to send back in JSON
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

        return jsonify({
            "prediction": pred_label,
            "confidence": pred_prob,
            "pdf_base64": pdf_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel requires the app to be exposed so it can handle the request
# This block is for local testing only
if __name__ == "__main__":
    app.run(debug=True, port=5000)
