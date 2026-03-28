import requests
import base64
import json

class LlamaVLMClient:
    def __init__(self, host="http://192.168.1.129", port="11434", model="llama3.2-vision"):
        self.url = f"{host}:{port}/api/chat" # O /api/chat dependiendo de tu backend (Ollama/vLLM)
        self.model = f"{model}"
    def _encode_image(self, image_path):
        """Convierte la imagen a Base64 para enviarla por JSON."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"[Error] Imagen no encontrada en: {image_path}")
            return None

    def query(self, system_policy, user_prompt, image_path):
        # Codificación de la imagen a Base64 (Requisito de Ollama)
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_policy # Aquí inyectamos tu política de toma de decisiones
                },
                {
                    "role": "user",
                    "content": user_prompt, # Aquí inyectamos el comando (ej: "Tengo hambre")
                    "images": [img_base64]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0 # Determinismo total para el Command Parser
            }
        }

        try:
            response = requests.post(self.url, json=payload, files="system_promt.txt", timeout=30)
            response.raise_for_status()
    
            # Para /api/chat, la respuesta está en ['message']['content']
            return response.json()['message']['content']
    
        except Exception as e:
            # Si da error, queremos ver el cuerpo del mensaje de Ollama
            if hasattr(e, 'response') and e.response is not None:
                return f"ERROR_API {e.response.status_code}: {e.response.text}"
            return f"ERROR_API: {str(e)}"
