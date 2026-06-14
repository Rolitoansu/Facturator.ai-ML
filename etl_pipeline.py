import pandas as pd
import json
import os
import shutil
import datetime

# Rutas de la Arquitectura Medallion
BRONZE_FILE = "data/bronze_feedback.jsonl"
SILVER_FILE = "data/silver_feedback.csv"
GOLD_FILE = "data/gold_training_data.csv"

def process_bronze_to_silver():
    """
    Lee los datos crudos (JSONL) de la capa Bronze, los limpia y los pasa a la capa Silver.
    """
    print("=== [MEDALLION: BRONZE -> SILVER] ===")
    if not os.path.exists(BRONZE_FILE) or os.stat(BRONZE_FILE).st_size == 0:
        print("No hay datos nuevos en la capa Bronze. Saliendo...")
        return None

    # 1. Ingesta de JSON a Pandas DataFrame
    records = []
    with open(BRONZE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    df_bronze = pd.DataFrame(records)
    print(f"Registros crudos cargados: {len(df_bronze)}")

    # 2. Data Cleaning
    # Eliminar duplicados exactos enviados en el mismo milisegundo por error
    df_silver = df_bronze.drop_duplicates(subset=['raw_text', 'correct_category'])
    
    # Limpiar strings (quitar saltos de línea y comillas que rompan el CSV)
    df_silver['raw_text'] = df_silver['raw_text'].astype(str).str.replace('\n', ' ', regex=False)
    df_silver['raw_text'] = df_silver['raw_text'].str.replace('"', '', regex=False)
    df_silver['raw_text'] = df_silver['raw_text'].str.replace(',', '', regex=False)
    
    # Filtrar datos de mala calidad (textos vacíos o categorías incorrectas)
    df_silver = df_silver[df_silver['raw_text'].str.len() > 5]
    df_silver = df_silver[df_silver['correct_category'].notna()]
    
    print(f"Registros tras limpieza (Silver): {len(df_silver)}")
    
    # 3. Guardar en capa Silver
    # Renombrar columnas para que coincidan con la estructura Gold
    df_silver = df_silver.rename(columns={'raw_text': 'text', 'correct_category': 'category'})
    df_silver[['text', 'category']].to_csv(SILVER_FILE, index=False)
    
    return df_silver

def process_silver_to_gold(df_silver):
    """
    Fusiona los datos limpios de Silver con la Master Table (Gold), asegurando la calidad.
    """
    print("\n=== [MEDALLION: SILVER -> GOLD] ===")
    
    # 1. Cargar la tabla maestra (Gold)
    if os.path.exists(GOLD_FILE):
        df_gold = pd.read_csv(GOLD_FILE, names=['text', 'category'], header=None) # No header in current gold
    else:
        df_gold = pd.DataFrame(columns=['text', 'category'])
        
    print(f"Registros en tabla maestra (Gold) actual: {len(df_gold)}")
    
    # 2. Fusión (Merge)
    df_new_gold = pd.concat([df_gold, df_silver[['text', 'category']]], ignore_index=True)
    
    # 3. Data Quality & Feature Engineering
    # Evitar tener la misma frase clasificada en dos categorías distintas por conflicto
    df_new_gold = df_new_gold.drop_duplicates(subset=['text'], keep='last') # Nos quedamos con la última corrección del usuario
    
    print(f"Registros tras fusión y deduplicación: {len(df_new_gold)}")
    
    # 4. Guardar
    df_new_gold.to_csv(GOLD_FILE, index=False, header=False, quoting=1) # quoting=1 ensures strings are wrapped in quotes
    print("Capa Gold actualizada con éxito.")

def archive_bronze_data():
    """
    Archiva los datos procesados para no volver a leerlos mañana.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"data/archive/bronze_feedback_{timestamp}.jsonl"
    os.makedirs("data/archive", exist_ok=True)
    shutil.move(BRONZE_FILE, archive_path)
    print(f"\n[ARCHIVE] Datos crudos movidos a: {archive_path}")

def retrain_ml_model():
    """
    Instancia el motor de ML para que consuma la nueva capa Gold y genere un nuevo .joblib
    """
    print("\n=== [MLOps: BATCH RETRAINING] ===")
    # Importamos el modelo aquí para evitar dependencias circulares
    from model import ReceiptMLModel
    
    # Al instanciarlo apuntando a Gold, y llamar a _train_category_classifier, sobrescribirá el caché
    print("Cargando arquitectura de ML y procesando capa Gold...")
    model = ReceiptMLModel(GOLD_FILE)
    model._train_category_classifier()
    print("¡Modelo reentrenado y guardado en caché! Listo para la producción.")

def run_pipeline():
    print("Iniciando ETL Pipeline (Arquitectura Medallion)...\n")
    df_silver = process_bronze_to_silver()
    
    if df_silver is not None and not df_silver.empty:
        process_silver_to_gold(df_silver)
        archive_bronze_data()
        retrain_ml_model()
        print("\n✅ Pipeline completada con éxito. Los servicios que lean model.joblib usarán el nuevo cerebro.")
    else:
        print("\n✅ Pipeline finalizada. Nada que procesar.")

if __name__ == "__main__":
    run_pipeline()
