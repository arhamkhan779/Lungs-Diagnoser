# 🫁 Lungs Diagnoser

A **Production-Level Multi-Model Diagnostic Agent** for lung disease detection. Built using Flask, TensorFlow, Scikit-learn, Langchain, and Gemini 2.0 Flash, this web application enables users to register, diagnose cancer probability, analyze chest X-rays, parse prescriptions, and generate reports — all in one platform.

![App Screenshot](https://github.com/arhamkhan779/Lungs-Diagnoser/blob/main/thumbnail.png?raw=true)


---

## 🚀 Features

### 🔐 User Registration & Session Support
- Unique user ID, name, email-based registration.
- Multi-user session handling.

### 🧠 Cancer Prediction Module
- Enter symptoms via form.
- Backend ML model predicts **Cancer vs. Non-Cancer probability**.

### 🩻 Segmentation & Classification from Chest X-rays
- **U-Net Segmentation**: Generates Lung Mask.
- Displays:
  - Original X-ray
  - Mask
  - Multiply Image
  - Overlay Image
- **Hierarchical Classification**:
  - Stage 1: Normal vs Abnormal
  - Stage 2: Abnormal subdivided into:
    - 🔸 Bacterial Pneumonia
    - 🔸 Viral Pneumonia
    - 🔸 COVID-19
    - 🔸 Tuberculosis

### 📄 Prescription Parser
- Upload old prescriptions.
- Gemini 2.0 Flash parses and formats the text for display.

### 📑 Report Generation Module
- Based on user input and diagnosis.
- Generate and download:
  - ✅ Medical Report
  - ✅ Diet Plan
  - ✅ Exercise Plan
- PDF download support.

### 📂 Data Storage
- Data stored inside `/data/{user_id}/`
  - Excel logs
  - Images (original, processed)
  - Reports and parsed content

---

## 🧩 Modules Overview

- `CancerPredictionModule`
- `SegmentationModule` (U-Net)
- `HierarchicalClassificationModule`
- `PrescriptionParserModule` (Gemini Flash)
- `ReportGenerationModule` (Medical, Diet, Exercise)
- `main_server.py` – Flask backend entry point

---

## 🛠️ Tech Stack

| Layer         | Tools / Libraries                                 |
|---------------|---------------------------------------------------|
| **Frontend**  | HTML, CSS, JavaScript                             |
| **Backend**   | Flask, Python, Langchain                          |
| **LLM**       | Gemini 2.0 Flash (Google)                         |
| **ML/DL**     | Scikit-learn, TensorFlow (U-Net, CNNs)            |
| **Image Ops** | OpenCV, Pillow                                    |
| **Deployment**| Session-based multi-user architecture             |


# 1. Clone the repository
git clone https://github.com/arhamkhan779/Lungs-Diagnoser.git
cd Lungs-Diagnoser

# 2. Create a .env file with your Gemini API Key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# 3. Set up a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python main_server.py
