import pandas as pd
import random
import string
import os

# Define categories and associated keywords
CATEGORIES = {
    "alimentacion": ["pollo", "leche", "pan", "fruta", "verdura", "carne", "pescado", "arroz", "huevos", "yogur", "queso", "agua", "refresco", "galletas", "cereal"],
    "transporte": ["billete", "gasolina", "diesel", "taxi", "uber", "cabify", "tren", "ave", "autobus", "bono", "mensual", "peaje", "parking", "estacionamiento", "vuelo"],
    "ropa": ["camisa", "pantalon", "zapatos", "chaqueta", "vestido", "falda", "calcetines", "ropa interior", "zapatillas", "deporte", "abrigo", "jersey", "cinturon", "bolso"],
    "ocio": ["cine", "palomitas", "restaurante", "cena", "comida", "menu", "tapas", "cerveza", "copa", "concierto", "entrada", "teatro", "museo", "hamburguesa", "pizza"],
    "salud": ["farmacia", "medicamento", "ibuprofeno", "paracetamol", "dentista", "optica", "gafas", "lentillas", "analisis", "clinica", "revision", "fisio", "masaje", "receta"],
    "hogar": ["ikea", "mueble", "estanteria", "silla", "mesa", "lampara", "bombilla", "pintura", "herramientas", "bricolaje", "factura", "luz", "agua", "internet", "fibra"],
    "suscripciones": ["netflix", "spotify", "amazon prime", "hbo", "disney", "gimnasio", "cuota", "mensualidad", "suscripcion", "revista", "prensa", "software", "licencia"]
}

# Define common merchants for each category
MERCHANTS = {
    "alimentacion": ["Mercadona", "Carrefour", "Lidl", "Aldi", "Dia", "Consum", "Ahorramas", "Eroski", "Alcampo"],
    "transporte": ["Renfe", "Uber", "Cabify", "Repsol", "Cepsa", "Galp", "Alsa", "Aena", "Metro Madrid", "TMB", "Empresa Municipal Transportes"],
    "ropa": ["Zara", "Pull and Bear", "Massimo Dutti", "Bershka", "Stradivarius", "Oysho", "H&M", "Primark", "Mango", "Decathlon", "El Corte Ingles"],
    "ocio": ["Cinesa", "Yelmo Cines", "Burger King", "McDonalds", "KFC", "Telepizza", "Dominos", "Starbucks", "100 Montaditos", "VIPS", "Ginos"],
    "salud": ["Farmacia", "Sanitas", "Adeslas", "Asisa", "Clinica Dental", "Multiopicas", "General Optica", "Visionlab"],
    "hogar": ["Ikea", "Leroy Merlin", "Bricomart", "Brico Depot", "Conforama", "Endesa", "Iberdrola", "Naturgy", "Canal de Isabel II", "Movistar", "Vodafone", "Orange"],
    "suscripciones": ["Netflix", "Spotify", "Amazon", "HBO Max", "Disney+", "Apple", "Google", "McFit", "Basic-Fit", "Altafit"]
}

def introduce_ocr_noise(text):
    """Introduces random OCR noise/typos into the text"""
    if random.random() > 0.4:
        return text # 60% chance of clean text
        
    noise_type = random.choice(["replace", "drop", "insert_space"])
    
    if noise_type == "replace" and len(text) > 3:
        idx = random.randint(1, len(text)-2)
        char = random.choice("0123456789!@#$%^&*")
        return text[:idx] + char + text[idx+1:]
    elif noise_type == "drop" and len(text) > 3:
        idx = random.randint(1, len(text)-2)
        return text[:idx] + text[idx+1:]
    elif noise_type == "insert_space" and len(text) > 3:
        idx = random.randint(1, len(text)-2)
        return text[:idx] + " " + text[idx:]
        
    return text

def generate_receipt_text(category, n_keywords=4):
    """Generates a synthetic receipt OCR string"""
    merchant = random.choice(MERCHANTS[category])
    merchant = introduce_ocr_noise(merchant.upper())
    
    keywords = random.sample(CATEGORIES[category], min(n_keywords, len(CATEGORIES[category])))
    
    # Add random filler words often found in receipts
    fillers = ["TOTAL", "EUR", "IVA", "FACTURA", "TICKET", "CAJERO", "FECHA", "TARJETA", "EFECTIVO", "CAMBIO", "GRACIAS POR SU VISITA"]
    
    lines = [
        merchant,
        f"CIF: {random.randint(10000000, 99999999)}{random.choice(string.ascii_uppercase)}",
        f"Fecha: {random.randint(1,28)}/{random.randint(1,12)}/2026",
        "----------------------"
    ]
    
    for word in keywords:
        price = round(random.uniform(1.0, 50.0), 2)
        lines.append(f"{introduce_ocr_noise(word.upper())}  {price}")
        
    lines.append("----------------------")
    total_price = round(random.uniform(10.0, 200.0), 2)
    lines.append(f"TOTAL EUR  {total_price}")
    lines.append(random.choice(fillers))
    
    # Add general noise
    final_text = "\n".join(lines)
    return introduce_ocr_noise(final_text)

def generate_dataset(n_samples_per_category=100, output_file="training_data.csv"):
    """Generates a full CSV dataset"""
    print(f"Generando dataset sintético de tickets ({n_samples_per_category} por categoría)...")
    
    data = []
    
    for category in CATEGORIES.keys():
        for _ in range(n_samples_per_category):
            text = generate_receipt_text(category, n_keywords=random.randint(2, 6))
            data.append({
                "text": text,
                "category": category
            })
            
    # Shuffle the dataset
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Dataset guardado en: {output_file} ({len(df)} ejemplos generados)")

if __name__ == "__main__":
    generate_dataset(n_samples_per_category=150, output_file="training_data.csv")
