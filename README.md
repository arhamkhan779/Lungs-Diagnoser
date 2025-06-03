
# 🫁 Lungs Diagnoser

A **Production-Level Multi-Model Diagnostic Agent** for lung disease detection. Built using Flask, TensorFlow, Scikit-learn, Langchain, and Gemini 2.0 Flash, this web application enables users to register, diagnose cancer probability, analyze chest X-rays, parse prescriptions, and generate reports — all in one platform.

![Home Dashboard](https://github.com/arhamkhan779/Lungs-Diagnoser/blob/main/thumbnail.png?raw=true)

---

## 🚀 Features

### 🔐 User Registration & Session Support
- Unique user ID, name, email-based registration.
- Multi-user session handling.

![Registration Page](https://github.com/arhamkhan779/Lungs-Diagnoser/blob/main/register.png?raw=true)

---

### 🧠 Cancer Prediction Module
- Enter symptoms via form.
- Backend ML model predicts **Cancer vs. Non-Cancer probability**.

![Cancer Prediction](https://github.com/arhamkhan779/Lungs-Diagnoser/blob/main/cancer_prediction.png?raw=true)

---

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

![X-ray Analysis](https://github.com/arhamkhan779/Lungs-Diagnoser/blob/main/X_ray_analysis.png?raw=true)

---

### 📄 Prescription Parser
- Upload old prescriptions.
- Gemini 2.0 Flash parses and formats the text for display.

![Prescription Parser](https://github.com/arhamkhan779/Lungs-Diagnoser/blob/main/prescription.png?raw=true)

---

### 📑 Report Generation Module
- Based on user input and diagnosis.
- Generate and download:
  - ✅ Medical Report
  - ✅ Diet Plan
  - ✅ Exercise Plan
- PDF download support.

![Generated Report](https://github.com/arhamkhan779/Lungs-Diagnoser/blob/main/report.png?raw=true)

---

### 📂 Data Storage
- All user data saved under `/data/{user_id}/`:
  - Symptom inputs (Excel)
  - Processed images (mask, overlay)
  - Prescription texts
  - Generated reports

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

---

## 🧪 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/arhamkhan779/Lungs-Diagnoser.git
cd Lungs-Diagnoser
````

### 2. Add Your Gemini API Key

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 3. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
python main_server.py
```

---

## 📄 License

MIT © [Arham Khan](https://github.com/arhamkhan779)


