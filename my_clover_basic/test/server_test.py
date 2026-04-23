import requests
import cv2
import numpy as np
import json
import time
import os
import base64
import sys

class VLM_Tester:
    def __init__(self, flask_url="http://localhost:5001"):
        self.url = flask_url
        self.endpoint = f"{self.url}/cmd_vel"
        
    def _generate_dummy_image(self):
        """Crea una imagen sintética para no depender de archivos externos"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, img_encoded = cv2.imencode('.jpg', img)
        return img_encoded.tobytes()

    def test_server_health(self):
        """Verifica si el servidor Flask responde (Heartbeat)"""
        try:
            response = requests.get(self.url)
            print(f"✅ Servidor detectado en {self.url}")
            return True
        except requests.exceptions.ConnectionError:
            print(f"❌ Error: El servidor Flask no está corriendo en {self.url}")
            return False

    def test_full_inference(self, query="Move forward", state="hover"):
        """
        Prueba el flujo completo: 
        Imagen -> Flask -> Ollama (Qwen) -> Respuesta
        """
        print(f"\n--- Iniciando Test de Inferencia (Modelo Qwen) ---")
        
        # Datos de prueba
        # 1. Cargar la imagen real
        image_path = "./test.png" # Usa la ruta absoluta si el script no está en la misma carpeta
        img = cv2.imread(image_path)

        # Validar que la imagen se ha leído correctamente
        if img is None:
            print(f"Error crítico: No se pudo encontrar o cargar '{image_path}'.")
            sys.exit(1)

        # 2. Preprocesamiento (Recomendado)
        # Redimensionar la imagen si es muy grande. Los modelos VL son muy pesados
        # y una imagen 4K puede ralentizar la inferencia o agotar la RAM.
        # Este código la reduce a un ancho máximo de 800px manteniendo la proporción.
        height, width = img.shape[:2]
        if width > 800:
            new_width = 800
            new_height = int((new_width / width) * height)
            img = cv2.resize(img, (new_width, new_height))

        # 3. Preparar la imagen para el envío (¡Sin Base64!)
        # Transformar la matriz de OpenCV a bytes (formato PNG) en memoria
        success, buffer = cv2.imencode('.png', img)
        if not success:
            print("Error crítico: Fallo al intentar codificar la imagen.")
            sys.exit(1)

        # Convertimos el buffer a bytes crudos, listos para adjuntar
        img_bytes = buffer.tobytes()

        topology = {
            "nodes": [{"id": 0, "pos": [0, 0]}],
            "edges": []
        }
        
        payload = {
            'query': query,
            'state': state,
            'topology': json.dumps(topology),
            'telemetry_text': "Alt: 1.5m, Battery: 85%"
        }
        
        # Usamos img_bytes en lugar de img_base64
        files = {
            'image': ('test.png', img_bytes, 'image/png')
        }

        start_time = time.time()
        try:
            response = requests.post(self.endpoint, data=payload, files=files)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ Inferencia exitosa en {elapsed_time:.2f} segundos")
                data = response.json()
                print(f"🤖 Respuesta del modelo: {data}")
                return data
            else:
                print(f"⚠️ Error en el servidor: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"❌ Error crítico durante la petición: {e}")
            return None

    def stress_test(self, iterations=5):
        """Ejecuta varias peticiones para medir estabilidad y tiempo promedio"""
        latencies = []
        print(f"\n--- Iniciando Stress Test ({iterations} iteraciones) ---")
        
        for i in range(iterations):
            start = time.time()
            res = self.test_full_inference()
            if res:
                latencies.append(time.time() - start)
        
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            print(f"\n--- Resultados del Stress Test ---")
            print(f"⏱️ Latencia promedio: {avg_lat:.2f}s")
            # Matemáticamente: $$Latency_{avg} = \frac{1}{n} \sum_{i=1}^{n} t_i$$

if __name__ == "__main__":
    # 1. Instanciar el tester (ajusta la IP si Flask está en otro PC)
    tester = VLM_Tester("http://localhost:5001")
    
    # 2. Comprobar si el servidor está vivo
    if tester.test_server_health():
        
        # 3. Probar una sola inferencia
        resultado = tester.test_full_inference(
            query="Describe your surroundings",
            state="Test"
        )
        
        # 4. Opcional: Probar rendimiento
        # tester.stress_test(iterations=3)