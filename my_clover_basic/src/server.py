#!/usr/bin/env python3.8
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import jsonpickle
from omegaconf import OmegaConf
from reasoning.VLMModel import ReasoningModel  # Ahora importa el Factory vLLM
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
vlm_model = None  

# --- FUNCIÓN DE DEPURACIÓN ZIP (Sin cambios, es útil) ---
def save_debug_zip(image_bytes, form_data, output_folder="debug_dumps"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(output_folder, f"payload_{timestamp}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        if image_bytes:
            zipf.writestr("captura_dron.jpg", image_bytes)
        zipf.writestr("datos_cliente.json", json.dumps(form_data, indent=4))
    logger.info(f"📦 Volcado guardado en: {zip_filename}")

@app.route('/cmd_vel', methods=['POST'])
def cmd_vel():
    global vlm_model 
    
    if vlm_model is None:
        return jsonify({'error': 'Modelo vLLM no inicializado en el servidor'}), 500

    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No se recibió imagen'}), 400

    raw_form = request.form.to_dict()
    
    # Desempaquetado del payload
    if 'payload' in raw_form:
        try:
            parsed_data = json.loads(raw_form['payload'])
        except json.JSONDecodeError:
            return jsonify({'error': 'Payload JSON inválido'}), 400
    else:
        parsed_data = raw_form

    query = parsed_data.get('query', '')
    state = parsed_data.get('state', 'Recognize Room')
    telemetry_text = parsed_data.get('telemetry_text', 'unknown')

    # Parseo de topología
    topology_raw = parsed_data.get('topology', {})
    topology_data = json.loads(topology_raw) if isinstance(topology_raw, str) else topology_raw

    # Procesamiento de imagen
    img_bytes = image_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_cv2 is None:
        return jsonify({'error': 'Imagen corrupta'}), 400

    try:
        logger.info(f"🧠 vLLM Inferencia | Estado: {state}")
        # El descriptor vLLM recibe la imagen cv2 directamente
        response = vlm_model.generate_trajectory(
            img_cv2, 
            query,    
            topology=topology_data, 
            state=state,
            telemetry_text=telemetry_text,
            test=parsed_data.get('test', False)
        )
        
        # vLLM devuelve un dict, lo codificamos para el cliente
        gpt_response = jsonpickle.encode(response)
        logger.info(f"✅ Respuesta generada: {response.get('movement', 'N/A')}")
        
        return Response(response=gpt_response, status=200, mimetype="application/json")

    except Exception as e:
        logger.error(f"❌ Error en motor vLLM: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--gpu-util', type=float, default=0.8) 
    args = parser.parse_args()

    # Overrides de configuración
    if args.model: cfg.model = args.model

    # 1. Resolver el '~' si existe en la configuración
    base_path = os.path.expanduser(cfg.get('model_path', '~/models'))

    # 2. Determinar qué modelo usar (prioridad al argumento de consola)
    model_name = args.model if args.model else cfg.model

    # 3. Concatenar para obtener la ruta absoluta final
    # Si args.model ya es una ruta absoluta, os.path.join es inteligente y la respeta
    full_model_path = os.path.join(base_path, model_name)

    logger.info(f"🚀 Cargando motor nativo vLLM: {cfg.model}")

    # Inicialización del modelo (Singleton)
    # IMPORTANTE: Eliminamos ollama_host y añadimos parámetros de vLLM
    try:
        vlm_model = ReasoningModel.create_model(
            cfg.model,
            max_tokens=cfg.get('max_tokens', 512),
            temperature=cfg.get('temperature', 0.0),
            gpu_memory_utilization=args.gpu_util,
            model_path=full_model_path # Ruta a los pesos .safetensors/bin
        )
    except Exception as e:
        logger.critical(f"💥 Error fatal cargando el modelo: {e}")
        exit(1)

    # use_reloader=False es OBLIGATORIO. 
    # Si Flask recarga, intentará cargar el modelo dos veces y dará Out of Memory.
    app.run(debug=True, host=args.host, port=args.port, use_reloader=False)