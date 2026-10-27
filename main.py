from deep_translator import GoogleTranslator
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import httpx
from database import engine, SessionLocal
import models as db_models

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# This automatically creates the farmer_data.db file and tables when the server starts
db_models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Crop Disease AI Backend")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for local development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 1. Load the Model on Startup (so it doesn't reload for every request)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Server starting. Using device: {device}")

with open('class_indices.json', 'r') as f:
    classes = json.load(f)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('best_crop_model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# 2. Define Image Preprocessing
def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

# 3. Create the Prediction Endpoint
@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    # Read the image sent from the mobile app
    image_bytes = await file.read()
    img_tensor = process_image(image_bytes)
    
    # Run Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    # Format the response as JSON
    results = []
    for i in range(3):
        results.append({
            "disease": classes[top3_catid[i]],
            "confidence": round(float(top3_prob[i]) * 100, 2)
        })
        
    return {
        "status": "success",
        "filename": file.filename,
        "predictions": results
    }

# --- FREE HOURLY WEATHER ENDPOINT (Open-Meteo) ---

@app.get("/weather")
async def get_weather(lat: float = 26.6528, lon: float = 92.7926):
    """
    Fetches a 100% free 5-hour weather forecast specifically for irrigation.
    Defaults to coordinates in Tezpur, Assam if none are provided.
    """
    # Open-Meteo requires no API key. 
    # Requesting temperature, humidity, rain chance, actual rain (mm), and soil moisture for exactly 5 hours.
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,rain,soil_moisture_0_to_7cm"
        f"&forecast_hours=5"
    )
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
    if response.status_code != 200:
        return {"status": "error", "message": "Failed to fetch weather data from Open-Meteo"}
        
    data = response.json()
    
    # Open-Meteo returns data in separate lists. We loop through them to organize the next 5 hours cleanly.
    hourly_data = data["hourly"]
    forecast_list = []
    
    for i in range(5):
        forecast_list.append({
            "datetime": hourly_data["time"][i],
            "temperature_celsius": hourly_data["temperature_2m"][i],
            "humidity_percent": hourly_data["relative_humidity_2m"][i],
            "precipitation_probability": hourly_data["precipitation_probability"][i],
            "rain_volume_mm": hourly_data["rain"][i],
            "soil_moisture": hourly_data["soil_moisture_0_to_7cm"][i]
        })
        
    return {
        "status": "success",
        "provider": "Open-Meteo (Free, No Key Required)",
        "irrigation_forecast": forecast_list
    }

# --- MASTER LOGIC ENGINE ENDPOINT ---

@app.post("/diagnose")
async def diagnose_and_advise(
    file: UploadFile = File(...),
    lat: float = 26.6528,
    lon: float = 92.7926,
    crop_type: str = "Unknown",
    lang: str = "en",
    db: Session = Depends(get_db)
):
    # 1. RUN THE ML MODEL
    image_bytes = await file.read()
    img_tensor = process_image(image_bytes)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    top_prob, top_catid = torch.topk(probabilities, 1)
    ai_raw_prediction = classes[top_catid[0]]
    confidence = round(float(top_prob[0]) * 100, 2)
    
    # SMART OVERRIDE LOGIC
    if "___" in ai_raw_prediction:
        predicted_plant, predicted_disease = ai_raw_prediction.split("___")
        if crop_type != "Unknown":
            detected_disease = f"{crop_type}___{predicted_disease}"
        else:
            detected_disease = ai_raw_prediction
    else:
        detected_disease = ai_raw_prediction

    # 2. FETCH THE WEATHER
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,rain,soil_moisture_0_to_7cm"
        f"&forecast_hours=5"
    )
    async with httpx.AsyncClient() as client:
        weather_resp = await client.get(weather_url)
        
    weather_data = weather_resp.json()
    hourly = weather_data["hourly"]
    
    # Calculate 5-hour averages
    avg_temp = sum(hourly["temperature_2m"]) / 5
    avg_humidity = sum(hourly["relative_humidity_2m"]) / 5
    total_rain = sum(hourly["rain"])
    avg_soil_moisture = sum(hourly["soil_moisture_0_to_7cm"]) / 5
    
    # 3. THE LOGIC ENGINE
    warnings = []
    
    if "blight" in detected_disease.lower():
        if avg_humidity > 80:
            warnings.append(f"CRITICAL: High humidity ({avg_humidity:.1f}%) will accelerate {detected_disease}. Apply fungicide.")
        else:
            warnings.append(f"Monitor closely: {detected_disease} detected, but current dry conditions will slow its spread.")
            
    if "healthy" in detected_disease.lower():
        warnings.append("Crop is healthy. No disease action required.")
        
    if total_rain > 0:
        warnings.append(f"IRRIGATION: {total_rain}mm of rain expected soon. Do not water.")
    elif avg_soil_moisture > 0.40:
        warnings.append("IRRIGATION: Soil moisture is adequate. No need to irrigate.")
    else:
        warnings.append("IRRIGATION: Soil is dry. Safe to run irrigation systems.")

    final_warning_string = " | ".join(warnings)

    # 4. SAVE TO DATABASE
    new_record = db_models.ScanRecord(
        disease_detected=detected_disease,
        confidence_score=confidence,
        temperature_c=avg_temp,
        humidity_percent=avg_humidity,
        soil_moisture=avg_soil_moisture,
        action_warning=final_warning_string
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    # --- 5. MULTILINGUAL TRANSLATION BLOCK ---
    final_disease = detected_disease.replace('_', ' ')
    final_warnings = warnings
    
    if lang != "en":
        try:
            print(f"--> Attempting to translate diagnosis to: {lang}")
            translator = GoogleTranslator(source='auto', target=lang)
            
            # Combine everything into one list to make a SINGLE request to Google
            texts_to_translate = [final_disease] + warnings
            translated_batch = translator.translate_batch(texts_to_translate)
            
            # Unpack the translated results
            final_disease = translated_batch[0]
            final_warnings = translated_batch[1:]
            
            print("--> Translation successful!")
        except Exception as e:
            print(f"--> Translation failed: {e}")
    # -----------------------------------------

    # 6. RETURN FINAL PACKAGE
    return {
        "status": "success",
        "record_id": new_record.id,
        "diagnosis": {
            "disease": final_disease,
            "confidence": f"{confidence}%",
            "original_ai_guess": ai_raw_prediction
        },
        "environmental_context": {
            "average_humidity": f"{avg_humidity:.1f}%",
            "average_temperature": f"{avg_temp:.1f}°C"
        },
        "farmer_instructions": final_warnings
    }
