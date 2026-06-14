import cv2
import numpy as np
import os

class ImageProcessor:
    """
    Pipeline de Computer Vision para mejorar la precisión del OCR en tickets.
    Utiliza OpenCV para limpiar sombras, arrugas y mejorar el contraste del texto.
    """
    
    @staticmethod
    def clean_receipt_image(image_path: str, output_path: str = None) -> str:
        """
        Lee una imagen, aplica transformaciones para resaltar el texto y la guarda.
        Si no se provee output_path, sobrescribe la imagen original.
        """
        if not output_path:
            output_path = image_path
            
        print(f"[{__name__}] Procesando imagen con OpenCV: {image_path}")
        
        # 1. Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen en {image_path}")
            
        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- AUTO-DESKEW (Corrección de Rotación) ---
        # Umbralización simple para encontrar el texto y calcular el ángulo
        _, temp_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(temp_thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            # Ajuste de ángulo según cómo minAreaRect devuelve valores
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Solo rotamos si es significativo
            if abs(angle) > 0.5 and abs(angle) < 45:
                print(f"[{__name__}] Auto-Deskew: Rotando imagen {angle:.2f} grados")
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Aplicar la rotación
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
        # --- CLAHE (Mejora de Contraste Local) ---
        # Ayuda a homogeneizar la iluminación si hay sombras del flash
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 3. Eliminar ruido manteniendo los bordes (Bilateral Filter / Gaussian)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 4. Umbralización adaptativa (Adaptive Thresholding)
        # Esto es clave para tickets porque la luz suele ser irregular.
        # Calcula el umbral para pequeñas regiones de la imagen en vez de una global.
        thresh = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # 5. Guardar la imagen limpia
        cv2.imwrite(output_path, thresh)
        
        print(f"[{__name__}] Imagen procesada y guardada en: {output_path}")
        return output_path
