#!/usr/bin/env python3.8
import multiprocessing as mp
import os
import sys
import logging
import argparse
import json
import zipfile
from datetime import datetime

# 1. CONFIGURACIÓN INICIAL DE MULTIPROCESAMIENTO
# Esto debe ejecutarse antes de que cualquier librería toque la GPU
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        # Variable de entorno para asegurar que vLLM use spawn internamente
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    except RuntimeError:
        pass

# 2. CONFIGURACIÓN DE LOGGING
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 3. VARIABLES GLOBALES (Se inicializarán en el bloque main)
app = None
vlm_model = None

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

def start_server():
    """Función principal que encapsula la carga y ejecución"""
    global vlm_model, app

    # 4. IMPORTS PEREZOSOS (Solo ocurren en el proceso principal)
    import cv2
    import numpy as np
    import jsonpickle
    from flask import Flask, Response, request, jsonify
    from omegaconf import OmegaConf
    from reasoning.VLMModel import ReasoningModel
    from transformers import Qwen2Tokenizer, Qwen2TokenizerFast

    # Parche de compatibilidad para Qwen
    if not hasattr(Qwen2Tokenizer, 'all_special_tokens_extended'):
        Qwen2Tokenizer.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
    if not hasattr(Qwen2TokenizerFast, 'all_special_tokens_extended'):
        Qwen2TokenizerFast.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

    # Argumentos de consola
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--gpu-util', type=float, default=0.8)
    args = parser.parse_args()

    # Cargar configuración de archivo
    cfg = OmegaConf.load("config.yaml")
    
    # Resolución de rutas del modelo
    base_path = os.path.expanduser(cfg.get('model_path', '~/models'))
    model_name = args.model if args.model else cfg.model
    full_model_path = os.path.join(base_path, model_name)

    logger.info(f"✅ Método de multiprocesamiento fijado en 'spawn'")
    logger.info(f"🚀 Cargando motor nativo vLLM: {model_name}")

    # Inicialización del modelo vLLM
    try:
        vlm_model = ReasoningModel.create_model(
            model_name,
            max_tokens=cfg.get('max_tokens', 512),
            temperature=cfg.get('temperature', 0.0),
            gpu_memory_utilization=args.gpu_util,
            model_path=full_model_path,
            # AÑADE ESTO SI TU CLASE LO PERMITE:
            trust_remote_code=True, 
            max_model_len=cfg.get('max_model_len', 32768) # Evita los 256k por defecto
        )
    except Exception as e:
        logger.critical(f"💥 Error fatal cargando el modelo: {e}")
        os._exit(1)

    # Inicialización de Flask
    app = Flask(__name__)

    @app.route('/cmd_vel', methods=['POST'])
    def cmd_vel():
        if vlm_model is None:
            return jsonify({'error': 'Modelo no inicializado'}), 500

        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No se recibió imagen'}), 400

        raw_form = request.form.to_dict()
        
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
        topology_raw = parsed_data.get('topology', {})
        topology_data = json.loads(topology_raw) if isinstance(topology_raw, str) else topology_raw

        # Procesamiento de imagen con OpenCV
        img_bytes = image_file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            return jsonify({'error': 'Imagen corrupta'}), 400

        try:
            logger.info(f"🧠 vLLM Inferencia | Estado: {state}")
            response = vlm_model.generate_trajectory(
                img_cv2, 
                query,    
                topology=topology_data, 
                state=state,
                telemetry_text=telemetry_text,
                test=parsed_data.get('test', False)
            )
            
            gpt_response = jsonpickle.encode(response)
            logger.info(f"Respuesta: {response}")
            logger.info(f" Movimiento propuesto: {response.get('movement', 'N/A')}")
            
            return Response(response=gpt_response, status=200, mimetype="application/json")

        except Exception as e:
            logger.error(f"❌ Error en motor vLLM: {e}")
            return jsonify({'error': str(e)}), 500

    # Ejecución del servidor
    # use_reloader=False evita que Flask cree un proceso duplicado que bloquee la GPU
    app.run(debug=False, host=args.host, port=args.port, use_reloader=False)

if __name__ == '__main__':
    start_server()