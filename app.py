import os
import shutil
import uuid
import requests
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import ReceiptMLModel
from etl_pipeline import run_pipeline

app = FastAPI(title="Facturator ML API")
model = ReceiptMLModel("data/gold_training_data.csv")

TEMP_DIR = "./temp_ml_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

class PredictRequest(BaseModel):
    image_url: str

class FeedbackRequest(BaseModel):
    raw_text: str
    correct_category: str

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": os.getenv("PORT", "8000")}

async def etl_cron_job():
    """
    Proceso Batch en Background:
    Ejecuta la pipeline de Data Engineering (Medallion) cada cierto tiempo sin bloquear la API.
    """
    while True:
        # En producción real esto podría ser 24 horas (24 * 60 * 60)
        # Para portfolio/demostración lo ponemos a 1 hora
        await asyncio.sleep(60 * 60) 
        print("\n[CronJob] 🕒 Iniciando proceso Batch de Data Engineering (ETL)...")
        try:
            # Ejecutar ETL en un thread separado para no bloquear FastAPI
            await asyncio.to_thread(run_pipeline)
            # Recargar el modelo en memoria con los nuevos datos
            model.reload_model()
        except Exception as e:
            print(f"[CronJob] ❌ Error en ETL Pipeline: {e}")

@app.on_event("startup")
async def startup_event():
    # Lanzar el CronJob como tarea asíncrona al arrancar el servidor
    asyncio.create_task(etl_cron_job())
    print("[FastAPI] 🚀 Servidor iniciado. ETL CronJob configurado en background.")

@app.post("/predict")
async def predict_receipt(request: PredictRequest):
    if not request.image_url:
        raise HTTPException(status_code=400, detail="image_url is required.")

    # Generamos un nombre temporal para el archivo
    temp_filename = f"{uuid.uuid4().hex}.jpg"
    temp_file_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        # Descargamos la imagen desde la URL de Supabase
        print(f"Downloading image from {request.image_url}...")
        response = requests.get(request.image_url, stream=True, timeout=15)
        response.raise_for_status()

        with open(temp_file_path, "wb") as buffer:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)
        from image_processor import ImageProcessor
        
        # 1. Aplicamos el pipeline de Computer Vision (OpenCV) para limpiar la imagen
        clean_image_path = ImageProcessor.clean_receipt_image(temp_file_path)
        
        # 2. Procesamos la imagen limpia con el modelo de ML
        result = model.predict(clean_image_path)
        print(f"ML Prediction complete: {result}")
        return result

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process receipt: {str(e)}")
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    if not request.raw_text or not request.correct_category:
        raise HTTPException(status_code=400, detail="raw_text and correct_category are required.")
        
    try:
        model.add_feedback(request.raw_text, request.correct_category)
        return {"status": "success", "message": "Feedback saved to Bronze layer. Awaiting ETL."}
    except Exception as e:
        print(f"Error adding feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to integrate feedback: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
