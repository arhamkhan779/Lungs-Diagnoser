from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
from utils import predict_cancers, process_segmentation_and_classification, extract_prescription_text
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from report_generation import DietPlanGenerator
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/register')
def home():
    try:
        error = request.args.get('error')
        return render_template('home.html', error=error)
    except Exception as e:
        raise e
    
@app.route('/predict_cancer')
def predict_cancer():
    try:
        user_id = request.args.get("user_id")  # 🔥 THIS is how you get it
        return render_template('predict_cancer.html', user_id=user_id)
    except Exception as e:
        raise e
    
@app.route('/segment_classify')
def segment_classify():
    try:
        user_id = request.args.get("user_id")
        return render_template('segment_classify.html', user_id=user_id)
    except Exception as e:
        raise e

@app.route('/upload_prescriptions')
def upload_prescriptions():
    try:
        user_id = request.args.get("user_id")
        return render_template('upload_prescription.html', user_id=user_id)
    except Exception as e:
        raise e

@app.route('/report_generation')
def report_generation():
    try:
        user_id = request.args.get("user_id")
        return render_template('report_generation.html', user_id=user_id)
    except Exception as e:
        raise e

@app.route("/diet_plan_generate", methods=["POST"])
def diet_plan_generator():
    try:
        data = request.get_json()
        user_id = data.get("user_id") if data else None
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        obj = DietPlanGenerator(id=user_id)
        plan = obj.generate_html_diet_plan()
        return jsonify({"response": plan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/medical_report_generate", methods=["POST"])
def medical_report_generator():
    try:
        data = request.get_json()
        user_id = data.get("user_id") if data else None
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        obj = DietPlanGenerator(id=user_id)
        plan = obj.generate_html_medical_report()
        return jsonify({"response": plan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/excercise_report_generate", methods=["POST"])
def excercise_report_generator():
    try:
        data = request.get_json()
        user_id = data.get("user_id") if data else None
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        obj = DietPlanGenerator(id=user_id)
        plan = obj.generate_html_exercise_plan()
        return jsonify({"response": plan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        name = request.form['name'].strip()
        email = request.form['email'].strip()
        user_id = request.form['id'].strip()

        if user_id in os.listdir("data"):
            # Redirect back to home with an error
            return redirect(url_for('home', error='User ID already exists'))

        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(user_folder, exist_ok=True)
        user_info_path = os.path.join(user_folder, 'user_info.xlsx')
        df = pd.DataFrame([{'Name': name, 'Email': email, 'ID': str(user_id)}])
        df.to_excel(user_info_path, index=False)

        return redirect(url_for('predict_cancer', user_id=user_id))
    except Exception as e:
        raise e

@app.route('/cancer_prediction', methods=['POST'])
def cancer_prediction():
    user_id = request.form['id'].strip()

    # Store each form input in a variable
    GENDER = request.form["GENDER"]
    AGE = int(request.form["AGE"])
    SMOKING = int(request.form["SMOKING"])
    YELLOW_FINGERS = int(request.form["YELLOW_FINGERS"])
    ANXIETY = int(request.form["ANXIETY"])
    PEER_PRESSURE = int(request.form["PEER_PRESSURE"])
    CHRONIC_DISEASE = int(request.form["CHRONIC_DISEASE"])
    FATIGUE = int(request.form["FATIGUE"])
    ALLERGY = int(request.form["ALLERGY"])
    WHEEZING = int(request.form["WHEEZING"])
    ALCOHOL_CONSUMING = int(request.form["ALCOHOL_CONSUMING"])
    COUGHING = int(request.form["COUGHING"])
    SHORTNESS_OF_BREATH = int(request.form["SHORTNESS_OF_BREATH"])
    SWALLOWING_DIFFICULTY = int(request.form["SWALLOWING_DIFFICULTY"])
    CHEST_PAIN = int(request.form["CHEST_PAIN"])

    # Call your function with individual arguments
    result = predict_cancers(
        GENDER,
        AGE,
        SMOKING,
        YELLOW_FINGERS,
        ANXIETY,
        PEER_PRESSURE,
        CHRONIC_DISEASE,
        FATIGUE,
        ALLERGY,
        WHEEZING,
        ALCOHOL_CONSUMING,
        COUGHING,
        SHORTNESS_OF_BREATH,
        SWALLOWING_DIFFICULTY,
        CHEST_PAIN
    )

    # Combine everything for saving
    combined_data = {
        "GENDER": GENDER,
        "AGE": AGE,
        "SMOKING": SMOKING,
        "YELLOW_FINGERS": YELLOW_FINGERS,
        "ANXIETY": ANXIETY,
        "PEER_PRESSURE": PEER_PRESSURE,
        "CHRONIC_DISEASE": CHRONIC_DISEASE,
        "FATIGUE": FATIGUE,
        "ALLERGY": ALLERGY,
        "WHEEZING": WHEEZING,
        "ALCOHOL_CONSUMING": ALCOHOL_CONSUMING,
        "COUGHING": COUGHING,
        "SHORTNESS_OF_BREATH": SHORTNESS_OF_BREATH,
        "SWALLOWING_DIFFICULTY": SWALLOWING_DIFFICULTY,
        "CHEST_PAIN": CHEST_PAIN,
        "Prediction": result["prediction"],
        "Cancer_Probability (%)": result["cancer_probability (%)"],
        "Non_Cancer_Probability (%)": result["non_cancer_probability (%)"]
    }

    # Save to Excel
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_folder, exist_ok=True)
    prediction_path = os.path.join(user_folder, "cancer_prediction.xlsx")
    pd.DataFrame([combined_data]).to_excel(prediction_path, index=False)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "prediction": f"{result['prediction']}\nCancer Probability: {result['cancer_probability (%)']}%\nNon-Cancer Probability: {result['non_cancer_probability (%)']}%"
        })
    else:
        return redirect(url_for('segment_classify', user_id=user_id))
    
@app.route('/segment_classify', methods=['POST'])
def segment_classify_post():
    user_id = request.form["id"].strip()
    file = request.files["file"]
    
    if file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_folder, exist_ok=True)

    # Save original uploaded image
    original_image_path = os.path.join(user_folder, "original_xray.png")
    file.save(original_image_path)

    # Run your segmentation/classification logic
    results = process_segmentation_and_classification(original_image_path)

    image_keys = ["input_image", "pred_mask", "multiply_image", "overlay"]
    pulmonary_prediction = results["pulmonary_prediction"]
    
    images_folder = os.path.join(user_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    # Save original image (optional)
    image_base64_data = {}
    for key in image_keys:
        image_array = results[key]
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        else:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        img_pil = Image.fromarray(image_array)

        # Save to BytesIO and encode to Base64
        buffer = BytesIO()
        img_pil.save(buffer, format="PNG")
        save_path = os.path.join(images_folder, f"{key}.png")
        img_pil.save(save_path)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_base64_data[key] = f"data:image/png;base64,{encoded_image}"

    # Optional: Save metadata to Excel
    metadata_df = pd.DataFrame([{
        "User ID": user_id,
        "Pulmonary Prediction": pulmonary_prediction
    }])
    excel_path = os.path.join(user_folder, "segment_classify.xlsx")
    metadata_df.to_excel(excel_path, index=False)

    # If called via JavaScript (AJAX)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "user_id": user_id,
            "pulmonary_prediction": pulmonary_prediction,
            "input_image": image_base64_data["input_image"],
            "pred_mask": image_base64_data["pred_mask"],
            "multiply_image": image_base64_data["multiply_image"],
            "overlay": image_base64_data["overlay"]
        })
    else:
        return redirect(url_for('upload_prescriptions', user_id=user_id))

@app.route('/upload_prescriptions', methods=['GET', 'POST'])
def upload_prescriptions_post():
    if request.method == 'POST':
        user_id = request.form.get("id", "").strip()
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(user_folder, exist_ok=True)

        files = request.files.getlist("files")
        if not files or len(files) == 0:
            return jsonify({"error": "No files selected"}), 400

        prescription_folder = os.path.join(user_folder, "prescription")
        os.makedirs(prescription_folder, exist_ok=True)

        saved_image_paths = []
        for idx, file in enumerate(files, start=1):
            ext = os.path.splitext(secure_filename(file.filename))[1] or ".png"
            image_filename = f"{user_id}_{idx}{ext}"
            image_path = os.path.join(prescription_folder, image_filename)
            file.save(image_path)
            saved_image_paths.append(image_path)

        extracted_text = extract_prescription_text(saved_image_paths)

        # Save to Excel
        metadata = {
            "User ID": user_id,
            "Extracted Text": extracted_text,
            "Image Paths": ", ".join(saved_image_paths)
        }
        df = pd.DataFrame([metadata])
        excel_path = os.path.join(user_folder, "prescription.xlsx")
        df.to_excel(excel_path, index=False)

        return jsonify({
            "extracted_text": extracted_text
        })

    # For initial GET request
    return render_template('upload_prescription.html', user_id=request.args.get('user_id', ''), extracted_text=None)

if __name__ == "__main__":
    app.run(debug=True)
