#!/usr/bin/env python3.8
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import jsonpickle
from omegaconf import OmegaConf
from reasoning.VLMModel import ReasoningModel
import json
import argparse
import logging
import os
import zipfile
from datetime import datetime

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 1. Cargar configuración base
cfg = OmegaConf.load("config.yaml")
cfg = OmegaConf.to_container(cfg, resolve=True)
cfg = OmegaConf.create(cfg)

app = Flask(__name__)
vlm_model = None  # Definimos el placeholder global

# --- FUNCIÓN DE DEPURACIÓN ZIP ---
def save_debug_zip(image_bytes, form_data, output_folder="debug_dumps"):
    """Crea un archivo .zip con la imagen y un JSON de los datos recibidos."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(output_folder, f"payload_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        if image_bytes:
            zipf.writestr("captura_dron.jpg", image_bytes)
        json_dump = json.dumps(form_data, indent=4)
        zipf.writestr("datos_cliente.json", json_dump)
        
    logger.info(f"📦 Volcado guardado en: {zip_filename}")
# ---------------------------------

@app.route('/cmd_vel', methods=['POST'])
def cmd_vel():
    global vlm_model 
    
    logger.info("📥 Petición recibida")
    
    if vlm_model is None:
        return jsonify({'error': 'Modelo no inicializado'}), 500

    # Extraer todos los datos del formulario para el ZIP
    form_data = request.form.to_dict()
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({'error': 'No se recibió imagen'}), 400

    query = form_data.get('query', '')
    topology_json = form_data.get('topology', '{}')
    state = form_data.get('state', '')
    telemetry_text = form_data.get('telemetry_text', '')

    # 1. Leer los bytes crudos enviados por el cliente
    img_bytes = image_file.read()

    # 2. Guardar el volcado de depuración
    save_debug_zip(img_bytes, form_data)

    # 3. Decodificar la imagen a una matriz OpenCV (Lo que ReasoningModel necesita)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_cv2 is None:
        return jsonify({'error': 'La imagen está corrupta o no es válida'}), 400

    try:
        logger.info(f"🧠 Invocando VLM con estado: {state}")
        # Llamada al modelo pasando img_cv2 (matriz NumPy), NO Base64
        response = vlm_model.generate_trajectory(
            img_cv2, 
            query,    
            topology=json.loads(topology_json), 
            state=state,
            telemetry_text=telemetry_text
        )
        
        logger.info("✅ Respuesta de Ollama obtenida")
        gpt_response = jsonpickle.encode(response)
        print("------------------------------------------")
        print(f"CONTENIDO REAL DE LA RESPUESTA: {response}")
        print("------------------------------------------")
        return Response(response=gpt_response, status=200, mimetype="application/json")

    except Exception as e:
        logger.error(f"❌ Error en el modelo: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--ollama-host', type=str, default=None)
    args = parser.parse_args()

    # 2. Aplicar overrides de terminal a la configuración
    if args.model: cfg.model = args.model
    if args.ollama_host: cfg.ollama_host = args.ollama_host

    logger.info(f"🚀 Iniciando modelo: {cfg.model}")
    logger.info(f"🌐 Conectando a Ollama en: {cfg.get('ollama_host')}")

    # 3. Inicializar el modelo globalmente UNA SOLA VEZ
    vlm_model = ReasoningModel.create_model(
        cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        ollama_host=cfg.get('ollama_host')
    )

    app.run(debug=True, host=args.host, port=args.port, use_reloader=False)