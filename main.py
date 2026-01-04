# main.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Optional: webcam capture
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# --------- Configuration ---------
DATA_PATH = "qt_dataset.csv"          # put your CSV here
MODEL_DIR = "model_outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "covid_model.joblib")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
FEATURES = ["Oxygen", "PulseRate", "Temperature"]
TARGET_COL = "Result"
# ---------------------------------

def ensure_model_exists():
    """Train and save model if not present, otherwise load it."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
        print(f"Loading existing model from '{MODEL_PATH}'")
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        return model, le

    # Train model
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Place 'qt_dataset.csv' in the same folder as this script.")
    print("Training model (dataset found). This may take a moment...")

    # Read CSV with latin1 fallback for weird encodings
    try:
        df = pd.read_csv(DATA_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding='latin1')

    # Validate columns
    missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {missing}. Required: {FEATURES + [TARGET_COL]}")

    # Keep only rows with non-null features + target
    df = df[FEATURES + [TARGET_COL]].dropna()

    X = df[FEATURES].astype(float)
    y_raw = df[TARGET_COL].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Train/test split for simple evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Quick evaluation printout
    y_pred = model.predict(X_test)
    print("Model trained. Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
    try:
        print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    except Exception:
        pass

    # Save model + label encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LE_PATH)
    print(f"Saved model to '{MODEL_PATH}' and label encoder to '{LE_PATH}'")

    return model, le

def capture_photo(save_path="captured_photo.jpg"):
    """Capture a single frame from default webcam and save it."""
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is not installed. Install opencv-python to use webcam capture.")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam.release()
        raise RuntimeError("Webcam could not be opened. Ensure a camera is connected and accessible.")
    print("Press 'Space' to capture, 'Esc' to cancel.")
    while True:
        ret, frame = cam.read()
        if not ret:
            cam.release()
            raise RuntimeError("Failed to read from webcam.")
        cv2.imshow("Capture - press Space to take photo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cam.release()
            cv2.destroyAllWindows()
            print("Capture cancelled.")
            return None
        if key == 32:  # Space
            # Save image
            cv2.imwrite(save_path, frame)
            cam.release()
            cv2.destroyAllWindows()
            print(f"Photo captured and saved to '{save_path}'")
            return save_path

def generate_pdf_report(vitals_dict, prediction_label, prob=None, photo_path=None, out_dir=".", patient_id=None, patient_name=None):
    """Generate a PDF report with vitals, prediction, timestamp, and optional photo."""
    # Generate filename as name_id.pdf
    if patient_name and patient_id:
        # Sanitize filename (remove special characters)
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in patient_name).strip().replace(' ', '_')
        safe_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in patient_id).strip()
        fname = f"{safe_name}_{safe_id}.pdf"
    else:
        # Fallback to timestamp if name/id not provided
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"prediction_report_{ts}.pdf"
    out_path = os.path.join(out_dir, fname)

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
        # prob expected to be float or dict; keep it simple
        try:
            pdf.cell(0, 7, f"Confidence (avg): {round(float(prob),3)}", ln=True)
        except Exception:
            pdf.cell(0, 7, f"Confidence: {prob}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 7, "Provided Vitals:", ln=True)
    for k, v in vitals_dict.items():
        pdf.cell(0, 6, f" - {k}: {v}", ln=True)
    pdf.ln(4)

    if photo_path and os.path.exists(photo_path):
        # Place image; leave space if too large
        try:
            pdf.image(photo_path, x=10, y=pdf.get_y(), w=80)
        except Exception as e:
            pdf.cell(0, 6, f"(Could not include photo: {e})", ln=True)

    # Footer / note
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 5, "Disclaimer: This is a prototype model output. This prediction should NOT be used as a medical diagnosis. Consult a medical professional for clinical decisions.")

    pdf.output(out_path)
    return out_path

def ask_for_vitals():
    """Prompt console input for vitals and return a dict using FEATURES keys."""
    print("Enter vitals (press Enter after each):")
    vit = {}
    for f in FEATURES:
        while True:
            val = input(f"{f}: ").strip()
            if val == "":
                print("Value required. Please enter a number.")
                continue
            try:
                # allow comma decimal
                val_norm = val.replace(",", ".")
                vit[f] = float(val_norm)
                break
            except ValueError:
                print("Invalid number. Try again.")
    return vit

def main():
    try:
        model, le = ensure_model_exists()
    except Exception as e:
        print("ERROR while preparing model:", e)
        return

    # Ask for patient name and ID first
    print("\n=== Patient Information ===")
    patient_name = input("Enter patient name: ").strip()
    while not patient_name:
        print("Patient name is required.")
        patient_name = input("Enter patient name: ").strip()
    
    patient_id = input("Enter patient ID: ").strip()
    while not patient_id:
        print("Patient ID is required.")
        patient_id = input("Enter patient ID: ").strip()

    # Ask whether to capture photo
    print("\n=== Photo Capture ===")
    photo_file = None
    use_cam = input("Do you want to capture a photo from webcam? (y/N): ").strip().lower()
    if use_cam == "y":
        if not OPENCV_AVAILABLE:
            print("OpenCV not available. Install with: pip install opencv-python")
        else:
            try:
                photo_file = capture_photo(save_path="captured_photo.jpg")
            except Exception as e:
                print("Photo capture failed:", e)
                photo_file = None

    # Ask for vitals
    print("\n=== Vital Signs ===")
    vitals = ask_for_vitals()

    # Build feature vector in required order
    fv = [vitals[k] for k in FEATURES]
    try:
        pred_idx = model.predict([fv])[0]
    except Exception as e:
        print("Prediction failed:", e)
        return

    # If label encoder is present, map back
    try:
        label_classes = le.classes_
        pred_label = label_classes[int(pred_idx)]
    except Exception:
        pred_label = str(pred_idx)

    # If classifier supports predict_proba, use average probability for chosen class
    pred_prob = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([fv])[0]
            pred_prob = float(np.max(probs))
    except Exception:
        pred_prob = None

    print("\n=== Prediction Result ===")
    print("Prediction:", pred_label)
    if pred_prob is not None:
        print("Confidence:", round(pred_prob, 4))

    # Generate PDF
    try:
        pdf_path = generate_pdf_report(vitals, pred_label, prob=pred_prob, photo_path=photo_file, out_dir=".", patient_id=patient_id, patient_name=patient_name)
        print(f"\nPDF report saved to: {pdf_path}")
    except Exception as e:
        print("Failed to generate PDF:", e)

if __name__ == "__main__":
    main()