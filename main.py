from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import os
from PIL import Image
import io

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
model = load_model("modelo_skinv3.h5")
print("Modelo cargado")

# Preprocesamiento de imagen
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Endpoint raíz
@app.get("/")
def read_root():
    return {"message": "API de diagnóstico funcionando"}

# Endpoint de predicción
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed = preprocess_image(contents)
        prediction = model.predict(processed)

        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        class_label = "Benigno" if class_idx == 0 else "Maligno"
        confidence_percent = f"{confidence:.2%}"

        # Umbral de seguridad diagnóstica
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
