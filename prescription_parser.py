import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class PrescriptionParser:
    '''
    Extracts medication and date from a list of prescription image paths using Gemini.
    '''

    def __init__(self, image_paths):
        if isinstance(image_paths, str):
            self.image_paths = [image_paths]
        elif isinstance(image_paths, list):
            self.image_paths = image_paths
        else:
            raise ValueError("image_paths must be a string or a list of strings.")

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-1.5-flash"
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def setup_prompt_combined(self, idx):
        return f"""
        You are a medical transcription expert.

        Your task is to analyze a handwritten prescription image and extract:
        1. The date of the prescription.
        2. The medication information.

        🎯 Expected Output:
        Return the extracted data in the following HTML structure:

        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 16px; margin: 20px 0;">
            <div style="background-color: #4CAF50; color: white; padding: 6px 12px; border-radius: 20px; display: inline-block; font-weight: bold; margin-bottom: 10px;">
            🧾 Prescription {idx} &nbsp;&nbsp; 📅 DATE_HERE
            </div>
            <table style="width:100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ccc; padding: 8px;">Medicine Name</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Dosage</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Frequency</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Duration</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                <td style="border: 1px solid #ccc; padding: 8px;">Panadol</td>
                <td style="border: 1px solid #ccc; padding: 8px;">500mg</td>
                <td style="border: 1px solid #ccc; padding: 8px;">Twice a day</td>
                <td style="border: 1px solid #ccc; padding: 8px;">5 days</td>
                </tr>
                <!-- Add more rows as needed -->
            </tbody>
            </table>
        </div>

        ✅ Notes:
        - Replace `DATE_HERE` with the date found in the prescription.
        - If no date is found, write "Not Found".
        - If any medicine-related field is missing, write "Not Mentioned".
        - Only return the final HTML output. Do not include any explanation.
        - only include html , code , dont include , ---html like this , only html code
        """

   

    @staticmethod
    def input_image_setup(image_path):
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_parts = [{
                "mime_type": "image/png" if image_path.endswith(".png") else "image/jpeg",
                "data": image_bytes
            }]
        return image_parts

    def get_gemini_response(self, input_text, image_parts, prompt):
        response = self.model.generate_content([input_text, image_parts[0], prompt])
        return response.text.strip()

    def process_all(self):
        result = ""
        for idx, image_path in enumerate(self.image_paths, start=1):
            try:
                image_parts = self.input_image_setup(image_path)
                full_prompt = self.setup_prompt_combined(idx)
                response_html = self.get_gemini_response("Extract date and medicine info.", image_parts, full_prompt)[7:-3]
                result += response_html + "\n"
            except Exception as e:
                result += f"❌ Error processing {image_path}: {e}\n"
        return result

