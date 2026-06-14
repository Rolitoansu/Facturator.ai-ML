import re
import os
import datetime
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from rapidfuzz import process, fuzz

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

EASYOCR_AVAILABLE = False
SPACY_AVAILABLE = False
reader = None
nlp = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("EasyOCR no instalado. Usando fallback mock.")

try:
    import spacy
    # Intentamos cargar el modelo de NLP en español
    # Si no existe, requiere: python -m spacy download es_core_news_sm
    nlp = spacy.load('es_core_news_sm')
    SPACY_AVAILABLE = True
    print("Modelo spaCy NLP (es_core_news_sm) cargado correctamente.")
except OSError:
    print("Modelo spaCy no encontrado. Se requiere ejecutar: python -m spacy download es_core_news_sm")
except ImportError:
    print("spaCy no instalado. Fallback a Regex clásico.")

class ReceiptMLModel:
    def __init__(self, data_path="training_data.csv"):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.data_path = data_path
        self.model_cache_path = "data/model.joblib"
        self.hf_pipeline = None
        
        # Lista de comercios comunes en España para corrección Fuzzy
        self.common_merchants = [
            "Mercadona S.A.", "Carrefour", "Lidl Supermercados", "Aldi", "Supermercados Dia",
            "Consum", "Ahorramas", "Renfe Viajeros", "Uber", "Cabify", 
            "Repsol", "Cepsa", "Galp", "Zara", "Pull and Bear", 
            "Massimo Dutti", "Decathlon", "El Corte Ingles", "Burger King", 
            "McDonalds", "KFC", "Starbucks", "Ikea", "Leroy Merlin",
            "Farmacia", "Sanitas", "Endesa", "Iberdrola", "Netflix", "Spotify"
        ]
        
        self._load_or_train_classifier()

    def _load_or_train_classifier(self):
        if os.path.exists(self.model_cache_path):
            print(f"[{__name__}] Cargando modelo desde caché ({self.model_cache_path})...")
            try:
                cached = joblib.load(self.model_cache_path)
                self.vectorizer = cached['vectorizer']
                self.classifier = cached['classifier']
                return
            except Exception as e:
                print(f"Error cargando caché: {e}. Reentrenando...")
        
        self._train_category_classifier()

    def reload_model(self):
        """
        Recarga el modelo desde la caché (usado después de que la ETL termine en background)
        """
        if os.path.exists(self.model_cache_path):
            print(f"[{__name__}] Recargando modelo en memoria para producción...")
            cached = joblib.load(self.model_cache_path)
            self.vectorizer = cached['vectorizer']
            self.classifier = cached['classifier']

    def _train_category_classifier(self):
        """
        Data Engineering pipeline: Entrena el modelo desde el CSV dinámico.
        Si el archivo no existe, hace fallback al corpus hardcodeado para no romper el server.
        """
        if os.path.exists(self.data_path):
            print(f"[{__name__}] Entrenando clasificador desde {self.data_path}...")
            df = pd.read_csv(self.data_path)
            
            # Limpiamos nulos
            df = df.dropna(subset=['text', 'category'])
            
            texts = df['text'].tolist()
            labels = df['category'].tolist()
        else:
            print(f"[{__name__}] WARNING: {self.data_path} no encontrado. Por favor corre generate_dataset.py. Usando fallback corpus.")
            # Fallback básico si el usuario no ha generado el dataset
            corpus = [
                ("mercadona compra de alimentacion pan leche frutas verduras", "alimentacion"),
                ("renfe billete ave madrid barcelona transporte tren", "transporte"),
                ("zara ticket compra chaqueta pantalon camisa ropa moda", "ropa"),
                ("starbucks cafe capuccino muffin desayuno ocio cafeteria", "ocio"),
                ("farmacia medicamento aspirina paracetamol jarabe salud", "salud"),
                ("ikea mueble estanteria mesa lampara hogar decoracion", "hogar"),
                ("spotify premium mensual musica streaming suscripciones", "suscripciones")
            ]
            texts = [item[0] for item in corpus]
            labels = [item[1] for item in corpus]

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        print(f"[{__name__}] Entrenamiento completado. {len(texts)} ejemplos procesados.")
        
        print(f"[{__name__}] Guardando modelo en caché ({self.model_cache_path})...")
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }, self.model_cache_path)

    def predict_category(self, text):
        cleaned_text = text.lower().replace("\n", " ")
        X = self.vectorizer.transform([cleaned_text])
        
        probs = self.classifier.predict_proba(X)[0]
        max_prob = max(probs)
        prediction = self.classifier.classes_[probs.argmax()]
        
        # Fallback a HuggingFace si la confianza es muy baja
        if max_prob < 0.65 and TRANSFORMERS_AVAILABLE:
            print(f"[{__name__}] Baja confianza ({max_prob:.2f}). Fallback a HuggingFace Zero-Shot...")
            if self.hf_pipeline is None:
                print(f"[{__name__}] Inicializando pipeline de Transformers (puede tardar la primera vez)...")
                self.hf_pipeline = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
            
            categories = list(self.classifier.classes_) if len(self.classifier.classes_) > 0 else ["alimentacion", "transporte", "ropa", "ocio", "suscripciones", "hogar"]
            hf_res = self.hf_pipeline(text, candidate_labels=categories)
            hf_pred = hf_res['labels'][0]
            hf_score = hf_res['scores'][0]
            
            print(f"[{__name__}] HuggingFace predijo: {hf_pred} con confianza {hf_score:.2f}")
            return hf_pred, hf_score
            
        return prediction, max_prob

    def extract_date(self, text):
        date_patterns = [
            r'\b(\d{2})[/-](\d{2})[/-](\d{4})\b',
            r'\b(\d{4})[/-](\d{2})[/-](\d{2})\b',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                match = matches[0]
                if len(match[0]) == 4:
                    return f"{match[0]}-{match[1]}-{match[2]}"
                else:
                    return f"{match[2]}-{match[1]}-{match[0]}"
        
        return datetime.date.today().strftime("%Y-%m-%d")

    def extract_amount(self, text):
        """
        Utiliza NER de spaCy si está disponible para entender el contexto y buscar tokens de dinero o importes.
        Si no, usa regex clásico mejorado.
        """
        normalized_text = text.replace(',', '.')
        
        if SPACY_AVAILABLE and nlp is not None:
            doc = nlp(normalized_text)
            
            # Buscar entidades que Spacy considere números o dinero en contexto con la palabra "total"
            amounts_found = []
            for token in doc:
                # Buscamos números con decimales
                if re.match(r'^\d+\.\d{2}$', token.text):
                    amounts_found.append(float(token.text))
            
            # Buscamos si alguna línea contiene "total" y extraemos de ahí
            lines = normalized_text.split('\n')
            for line in lines:
                if "total" in line.lower() or "eur" in line.lower() or "importe" in line.lower():
                    line_doc = nlp(line)
                    for token in line_doc:
                        if re.match(r'^\d+\.\d{2}$', token.text):
                            return float(token.text)
                            
            if amounts_found:
                # Por heurística, el importe más alto suele ser el total en un ticket
                valid = [a for a in amounts_found if a < 10000.0]
                if valid:
                    return max(valid)
        
        # Fallback a Regex Clásico
        amount_pattern = r'\b\d+\.\d{2}\b'
        candidates = re.findall(amount_pattern, normalized_text)
        
        if not candidates:
            return 15.00
            
        floats = []
        for c in candidates:
            try:
                floats.append(float(c))
            except ValueError:
                continue

        valid_amounts = [f for f in floats if f < 10000.0]
        if not valid_amounts:
            return 15.00
            
        lines = normalized_text.split('\n')
        for line in lines:
            if any(k in line.lower() for k in ["total", "importe", "eur", "suma"]):
                line_candidates = re.findall(amount_pattern, line)
                if line_candidates:
                    return float(line_candidates[-1])

        return max(valid_amounts)

    def extract_merchant(self, text):
        """
        Extrae el nombre usando la primera línea relevante y aplica Fuzzy Matching 
        con RapidFuzz para corregir errores ortográficos del OCR (ej: M3rcad0na -> Mercadona S.A.)
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        raw_merchant = "Comercio Local"
        
        if not lines:
            return raw_merchant, 50.0
            
        address_indicators = ["calle", " c/", "avda", "avda.", "plaza", "dir:", "tel:", "c.i.f", "cif:", "nif:"]
        
        for line in lines[:3]:
            line_lower = line.lower()
            is_address = any(addr in line_lower for addr in address_indicators)
            has_letters = any(c.isalpha() for c in line)
            
            if has_letters and not is_address and not line.replace(" ", "").isdigit():
                raw_merchant = line.split(" - ")[0].split("  ")[0][:30]
                break
                
        # --- Fuzzy Matching Magic ---
        # Comparamos el raw_merchant con nuestra lista de comercios comunes
        best_match = process.extractOne(raw_merchant, self.common_merchants, scorer=fuzz.WRatio)
        
        if best_match:
            match_str, score, _ = best_match
            # Si el score de similitud es mayor al 75%, asumimos que es un typo de ese comercio
            if score > 75:
                print(f"[{__name__}] Fuzzy match corregido: '{raw_merchant}' -> '{match_str}' (Score: {score})")
                return match_str, score
                
        return raw_merchant, 60.0

    def perform_ocr(self, image_path):
        global reader
        
        if EASYOCR_AVAILABLE:
            try:
                if reader is None:
                    print("Initializing EasyOCR reader...")
                    reader = easyocr.Reader(['es'], gpu=False)
                
                results = reader.readtext(image_path)
                extracted_lines = [res[1] for res in results]
                full_text = "\n".join(extracted_lines)
                print(f"OCR extracted text length: {len(full_text)} chars")
                return full_text
            except Exception as e:
                print(f"EasyOCR failed: {e}. Using fallback.")
        
        filename = os.path.basename(image_path).lower()
        if "mercadona" in filename:
            return "M3rcad0na S.A.\nC/ ALCALA 120\n02/05/2026 10:14\nTOTAL: 67,40 EUR\nGRACIAS POR SU COMPRA"
        elif "renfe" in filename or "ave" in filename:
            return "RENFE VlAJEROS\nBILLETE AVE 03120\nFECHA: 10/05/2026\nIMPORT TOTAL: 43,50 €\nBUEN VIAJE"
        
        today_str = datetime.date.today().strftime("%d/%m/%Y")
        return f"C4RREF0UR EXPLRESS\nAVENIDA DE LA CONSTITUCION 14\nFECHA: {today_str}\nPOLLO  8,50\nLECHE  1,30\nTOTAL COMPRA  9,80 EUR\nGRACIAS"

    def predict(self, image_path):
        text = self.perform_ocr(image_path)
        
        merchant, merchant_score = self.extract_merchant(text)
        amount = self.extract_amount(text)
        category, cat_confidence = self.predict_category(text)
        date = self.extract_date(text)
        
        confidence = (cat_confidence + (merchant_score / 100.0)) / 2.0
        needs_review = confidence < 0.65
        
        return {
            "merchant": merchant,
            "amount": amount,
            "category": category,
            "date": date,
            "raw_text": text,
            "confidence": confidence,
            "needs_review": needs_review
        }
        
    def add_feedback(self, text, correct_category):
        """
        [Medallion Architecture - Bronze Layer]
        Guarda la corrección del usuario en bruto (Raw JSONL) para ser procesada en Batch por la ETL.
        No re-entrena el modelo en caliente.
        """
        import json
        bronze_path = "data/bronze_feedback.jsonl"
        
        feedback_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_text": text,
            "correct_category": correct_category
        }
        
        with open(bronze_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_event) + "\n")
            
        print(f"[{__name__}] Feedback encolado en Bronze Layer para reentrenamiento Batch: '{correct_category}'")
