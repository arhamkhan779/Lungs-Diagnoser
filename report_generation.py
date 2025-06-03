import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
load_dotenv()



class DietPlanGenerator:
    '''
    python module to generate a 
    specific diet plan for the patient
    infected with different lung diseases
    '''
    root = "data"
    def __init__(self,id):
        self.user_id = id
        self.user_data = os.path.join(DietPlanGenerator.root,id)
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model = "gemini-2.0-flash"
        self.user_info = os.path.join(self.user_data,"user_info.xlsx")
        self.cancer_predictions = os.path.join(self.user_data,"cancer_prediction.xlsx")
        self.x_ray_results = os.path.join(self.user_data,"segment_classify.xlsx")
        self.past_prescription_data = os.path.join(self.user_data,"prescription.xlsx")
    
    def get_user_info(self):
        df = pd.read_excel(self.user_info)
        return {
            "user_name":df.Name.values[0],
            "user_id":int(df.ID.values[0]) ,
            "user_email":df.Email.values[0]
        }
    
    def get_cancer_prediction_results(self):
        data = {}
        df = pd.read_excel(self.cancer_predictions)

        for column in df.columns:
            value = df[column].values[0]
            if column == "Prediction":
                data[column] = str(value)
            elif column == "GENDER":
                if int(df['GENDER'].values[0]) == 0:
                    data['GENDER'] = "Male"
                else:
                    data["GENDER"] = "Female"
            else:
                if int(df[column].values[0]) == 0:
                    data[column] = "No"
                elif int(df[column].values[0]) == 1:
                    data[column] = "Yes"
                else:
                    data[column] = int(df[column].values[0])
        return data
    
    def get_segment_classify_results(self):
        data = {}
        df = pd.read_excel(self.x_ray_results)
        pulmonary_prediction_results = df['Pulmonary Prediction'].values[0]
        data['X_ray_results']=pulmonary_prediction_results
        
        images_folder = os.path.join(self.user_data,"images")
        for images in os.listdir(images_folder):
            data[images] = os.path.join(images_folder,images)
        return data
    
    def get_past_prescription_data(self):
        data = {}
        df= pd.read_excel(self.past_prescription_data)
        data["Extracted_text"] = df['Extracted Text'].values[0]
        return data['Extracted_text']
    
    def load_model(self):
        model = ChatGoogleGenerativeAI(model=self.model,api_key=self.api_key)
        return model 
    
    def initialize_medical_report_prompt(self, patient_data, cancer_prediction_results, x_ray_analysis_results, past_prescriptions):
        prompt = f"""
        You are a highly skilled AI medical assistant and expert in responsive web design.

        🎯 Objective:
        Generate a **professional, clean HTML report** titled "Comprehensive Medical Report" for a patient undergoing evaluation for lung-related conditions.

        🔹 Include the following in the HTML:
        1. **Header Title**:
            - Centered, bold, large text: "Comprehensive Medical Report"

        2. **Patient Summary Card**:
            - Patient ID
            - Name
            - Email
            - Gender

        3. **Diagnostic Summary Section**:
            - Cancer Prediction Results:
                - Present as a clean bullet list or table
            - X-Ray Analysis:
                - Present a summary of AI analysis results in readable format
            - Past Prescriptions:
                - Show past medication or treatment summaries in bullet points or a bordered box

        4. **Styling Requirements**:
            - Use **inline CSS only**
            - Include **Google Fonts** like 'Poppins' or 'Open Sans'
            - Use **light theme colors**: white background, shades of blue and gray
            - Include **rounded corners**, **box shadows**, and **ample spacing**
            - Make layout **responsive for both desktop and mobile**
            - Ensure print/PDF readability
            - Use icons like 📄 🩻 💊 for section headers to enhance clarity

        👤 Patient Information:
        Name: {patient_data['user_name']}
        ID: {patient_data['user_id']}
        Email: {patient_data['user_email']}
        Gender: {cancer_prediction_results.get('GENDER', 'Unknown')}

        🧬 Cancer Prediction Results (for display):
        {cancer_prediction_results}

        🩻 X-ray Analysis Results (summary only):
        {x_ray_analysis_results}

        💊 Past Prescription Summary:
        {past_prescriptions}

        ❗Only return a complete and well-structured HTML document with all content and inline CSS.
        """
        return prompt

    
    def initialize_prompt(self, patient_data, cancer_prediction_results, x_ray_analysis_results, past_prescriptions):

        prompt = f"""
        You are a professional AI health assistant and skilled UI designer.

        🎯 Objective:
        Generate a **visually stunning, medically-informed HTML page** for a **Personalized 30-Day Diet Plan** for a patient with lung-related health conditions.

        Use the provided patient background (cancer prediction results, x-ray findings, and prescription history) **only as internal context** to tailor the diet plan—but do **NOT** display any of this medical data in the HTML output.

        🔹 HTML Page Requirements:
        - A **centered, modern title** in dark navy blue: "Personalized 30-Day Diet Plan"
        - A **summary card** at the top with basic patient info:
            - Patient ID
            - Name
            - Email
            - Gender
        - A **weekly breakdown** of the 30-day diet plan:
            - Show diet plan in 4 weekly sections (Week 1 to Week 4)
            - For each day, include:
                - 🥣 Breakfast
                - 🍛 Lunch
                - 🍽️ Dinner
                - 📝 Notes (optional)
        - A final **recommendation section** that includes:
            - ✅ Dietary Recommendations (e.g., hydration, anti-inflammatory foods)
            - ⚠️ Caution Notes (e.g., avoid smoking, reduce salt intake)

        🎨 Styling Requirements:
        - Use **inline CSS only**
        - Include **Google Fonts** (like 'Poppins' or 'Open Sans') for a clean, modern look
        - Use **cards, tables with alternating row colors**, and rounded corners
        - Add light **shadows, padding**, and **professional spacing**
        - Make the layout **mobile-friendly and responsive**
        - Use a consistent **light theme with shades of blue, gray, and white**
        - Add icons like 🥗 🍲 🍎 to make the design more engaging

        👤 Patient Information:
        Name: {patient_data['user_name']}
        ID: {patient_data['user_id']}
        Email: {patient_data['user_email']}
        Gender: {cancer_prediction_results.get('GENDER', 'Unknown')}

        🧬 Medical Context (for internal reasoning only – do not include in HTML output):
        Cancer Prediction Results: {cancer_prediction_results}
        X-Ray Analysis Results: {x_ray_analysis_results}
        Past Prescriptions Summary: {past_prescriptions}

        ❗ Only return a full, clean HTML page with inline CSS.
        """
        return prompt
 
    
    def initialize_exercise_plan_prompt(self, patient_data, cancer_prediction_results, x_ray_analysis_results, past_prescriptions):
        prompt = f"""
        You are a highly skilled AI fitness coach and expert in responsive web design.

        🎯 Objective:
        Generate a **professional, clean HTML report** titled "30-Day Personalized Exercise Plan" tailored for a patient based on their medical analysis results.

        🔹 Include the following in the HTML:
        1. **Header Title**:
            - Centered, bold, large text: "30-Day Personalized Exercise Plan"

        2. **Patient Summary Card**:
            - Patient ID
            - Name
            - Email
            - Gender

        3. **Medical Analysis Summary**:
            - Cancer Prediction Results: Present key points in bullet format or table
            - X-Ray Analysis: Summarize AI analysis results in simple language
            - Past Prescriptions: List past medications or treatments in bullet points or a bordered box

        4. **Exercise Plan Section**:
            - Generate a detailed 30-day exercise plan
            - Exercises should be safe and appropriate considering the medical analysis above
            - Present daily workouts with exercise names, duration/reps, rest days
            - Use bullet lists or tables for clarity

        5. **Styling Requirements**:
            - Use **inline CSS only**
            - Include **Google Fonts** like 'Poppins' or 'Open Sans'
            - Use **light theme colors**: white background, shades of green and gray
            - Include **rounded corners**, **box shadows**, and **ample spacing**
            - Make layout **responsive for desktop and mobile**
            - Ensure print/PDF readability
            - Use icons like 🏃‍♂️ 💪 🧘‍♀️ for section headers to enhance clarity

        👤 Patient Information:
        Name: {patient_data['user_name']}
        ID: {patient_data['user_id']}
        Email: {patient_data['user_email']}
        Gender: {cancer_prediction_results.get('GENDER', 'Unknown')}

        🧬 Cancer Prediction Results:
        {cancer_prediction_results}

        🩻 X-ray Analysis Results:
        {x_ray_analysis_results}

        💊 Past Prescription Summary:
        {past_prescriptions}

        ❗Only return a complete and well-structured HTML document with all content and inline CSS.
        """
        return prompt


    def generate_html_diet_plan(self):
        # Step 1: Load all inputs
        patient_data = self.get_user_info()
        cancer_data = self.get_cancer_prediction_results()
        xray_data = self.get_segment_classify_results()
        past_prescriptions = self.get_past_prescription_data()

        # Step 2: Create the prompt
        prompt_text = self.initialize_prompt(
            patient_data,
            cancer_data,
            xray_data,
            past_prescriptions
        )

        # Step 3: Generate with Gemini
        model = self.load_model()
        response = model.invoke(prompt_text)

        return response.content[7:-3]
    
    def generate_html_medical_report(self):
        # Step 1: Load all inputs
        patient_data = self.get_user_info()
        cancer_data = self.get_cancer_prediction_results()
        xray_data = self.get_segment_classify_results()
        past_prescriptions = self.get_past_prescription_data()

        # Step 2: Create the prompt for the medical report
        prompt_text = self.initialize_medical_report_prompt(
            patient_data,
            cancer_data,
            xray_data,
            past_prescriptions
        )

        # Step 3: Generate HTML using Gemini (or any other model)
        model = self.load_model()
        response = model.invoke(prompt_text)

        # Step 4: Clean and return the generated HTML
        return response.content[7:-3]

    def generate_html_exercise_plan(self):
        # Step 1: Load all inputs
        patient_data = self.get_user_info()
        cancer_data = self.get_cancer_prediction_results()
        xray_data = self.get_segment_classify_results()
        past_prescriptions = self.get_past_prescription_data()

        # Step 2: Create the prompt for the exercise plan
        prompt_text = self.initialize_exercise_plan_prompt(
            patient_data,
            cancer_data,
            xray_data,
            past_prescriptions
        )

        # Step 3: Generate HTML using Gemini (or any other model)
        model = self.load_model()
        response = model.invoke(prompt_text)

        # Step 4: Clean and return the generated HTML
        return response.content[7:-3]


