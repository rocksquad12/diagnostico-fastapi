from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import uvicorn
import cv2
import os

app = FastAPI()

# Permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
print("Cargando modelo...")
model = load_model("modelo_clasificador_skin_lesions.h5")
print("Modelo cargado")

# Preprocesamiento de imagen con OpenCV
def preprocess_image(img_bytes):
    # Guardar imagen temporalmente
    temp_path = "temp_received.jpg"
    with open(temp_path, "wb") as f:
        f.write(img_bytes)
    
    img = cv2.imread(temp_path)
    if img is None:
        raise ValueError("La imagen no pudo ser cargada con OpenCV.")

    img_resized = cv2.resize(img, (224, 224))  # Redimensionar
    img_preprocessed = preprocess_input(img_resized)  # Normalizar como EfficientNet
    return np.expand_dims(img_preprocessed, axis=0)

# Endpoint raíz
@app.get("/")
def read_root():
    return {"message": "API de diagnóstico funcionando"}

# Endpoint de predicción
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Guardar la imagen original para depurar si es necesario
        with open("debug_received.jpg", "wb") as f:
            f.write(contents)

        processed = preprocess_image(contents)
        prediction = model.predict(processed)

        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        class_label = "Benigno" if class_idx == 0 else "Maligno"
        confidence_percent = f"{confidence:.2%}"

        umbral_confianza = 0.60
        warning_msg = None
        if confidence < umbral_confianza:
            warning_msg = (
                "Confianza baja en el diagnóstico. Se recomienda una revisión médica adicional."
            )

        return {
            "prediction": {
                "label_id": class_idx,
                "label_name": class_label,
                "confidence": confidence_percent
            },
            "status": "success",
            "message": "Diagnóstico realizado correctamente.",
            "warning": warning_msg
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )

# Ejecutar localmente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
