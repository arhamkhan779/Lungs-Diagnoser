import cv2 
from tensorflow import keras 
import numpy as np 


model_path = r"models\classification_model.keras"
def Load_Model(model_path):
     model = keras.models.load_model(model_path)
     return model 

print(f"Loading Classification Model")
model = Load_Model(model_path)

def classify_disease(Image_path):
        Super_Class=['Normal','Abnormal']
        Subclasses=['Viral Pneumonia','Bacterial Pneumonia','Corona Virus','Tuberclosis']
        image=cv2.imread(Image_path)
        img=cv2.resize(image,(64,64))
        img=img/255
        img=img.reshape(1,64,64,3)
        y_pred_binary, y_pred_subclass = model.predict(img)

        y_pred_binary = (y_pred_binary > 0.5).astype(int).flatten()
        y_pred_subclass = np.argmax(y_pred_subclass, axis=1) 
        if y_pred_binary[0] == 0:         
            output=f"Super Class Output : {Super_Class[y_pred_binary[0]]} Subclass Output: NONE"
            
        else:
            output=f"Super Class Output : {Super_Class[y_pred_binary[0]]} Subclass Output: {Subclasses[y_pred_subclass[0]]}"
        return output