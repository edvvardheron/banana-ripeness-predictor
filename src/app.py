import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware # <--- Import this

app = FastAPI(title="Banana Ripeness Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# ----------------------

# Path to the model we just trained
MODEL_PATH = "models/banana_mobilenet.keras"
IMAGE_SIZE = (299, 299) # Must match training size

# Load the model into memory once when the server starts
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

def preprocess_image(image_bytes):
    """
    Transforms raw bytes into the format the model expects.
    1. Open as PIL image
    2. Convert to RGB (removes alpha channel if present)
    3. Resize to 224x224
    4. Normalize to [0, 1]
    5. Expand dimensions to (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Banana Ripeness API is online. Use the /predict endpoint."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server."}

    try:
        # 1. Read the uploaded file
        contents = await file.read()
        
        # 2. Preprocess
        processed_image = preprocess_image(contents)
        
        # 3. Predict
        # .predict() returns a 2D array: [[prediction]]
        prediction = model.predict(processed_image)
        days_result = float(prediction[0][0])
        
        # 4. Human-readable logic
        status = "Ready to eat!" if days_result < 1.0 else f"Wait about {round(days_result)} more days."
        
        return {
            "filename": file.filename,
            "prediction_days": round(days_result, 2),
            "status": status,
            "model_version": "MobileNetV2-Transfer"
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

if __name__ == "__main__":
    # Run the server locally on port 8000
    uvicorn.run(app, host="localhost", port=8000)