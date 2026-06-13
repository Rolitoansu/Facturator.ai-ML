import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from model import ReceiptMLModel

app = FastAPI(title="Facturator ML API")
model = ReceiptMLModel()

TEMP_DIR = "./temp_ml_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": os.getenv("PORT", "5000")}

@app.post("/predict")
async def predict_receipt(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are supported.")

    temp_file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = model.predict(temp_file_path)
        print(f"ML Prediction complete for {file.filename}: {result}")
        return result

    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process receipt: {str(e)}")
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
