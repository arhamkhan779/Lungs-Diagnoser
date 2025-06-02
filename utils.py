from werkzeug.utils import secure_filename
from prescription_parser_module import PrescriptionParser
from cancer_diagnoser import predict_lung_disease
from classification_module import classify_disease
from segmentation_module import mask_overlay_original




def extract_prescription_text(image_paths):
    """
    Takes a list of image file paths and returns extracted text from all prescriptions.
    """
    if not image_paths:
        return "No image paths provided"

    try:
        parser = PrescriptionParser(image_paths)
        extracted_text = parser.process_all()
        return extracted_text
    except Exception as e:
        return f"Error during extraction: {str(e)}"

def predict_cancers(
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
):
    # Call the prediction function, assuming it returns a dict like:
    # { "prediction": ..., "cancer_probability (%)": ..., "non_cancer_probability (%)": ... }
    result = predict_lung_disease(
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
    
    #
    response = {
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
        "prediction": result["prediction"],
        "cancer_probability (%)": result["cancer_probability (%)"],
        "non_cancer_probability (%)": result["non_cancer_probability (%)"]
    }
    return response

def process_segmentation_and_classification(image_path):
    # Get segmented images (numpy arrays)
    input_image, pred_mask, multiply_image, overlay = mask_overlay_original(image_path)

    # Run classification on multiply_image
    pulmonary_result = classify_disease(input_image)

    return {
        "input_image": input_image,
        "pred_mask": pred_mask,
        "multiply_image": multiply_image,
        "overlay": overlay,
        "pulmonary_prediction": pulmonary_result
    }


