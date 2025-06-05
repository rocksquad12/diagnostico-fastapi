from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import os
from PIL import Image
import io

app = FastAPI()

# Permitir CORS (para conectar con frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto si deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo .h5
print("Cargando modelo...")
model = load_model("modelo_skinv3.h5")
print("Modelo cargado")

# Función para preprocesar imagen
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Ajusta según tu modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normaliza si tu modelo lo necesita
    return img_array

# Endpoint de prueba
@app.get("/")
def read_root():
    return {"message": "API de diagnóstico funcionando"}

# Endpoint para predecir
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed = preprocess_image(contents)
    prediction = model.predict(processed)
    
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "class": class_idx,
        "confidence": confidence
    }

# Ejecutar localmente si se desea
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
