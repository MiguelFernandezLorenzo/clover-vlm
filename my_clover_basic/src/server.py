#!/usr/bin/env python3.8
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from werkzeug.utils import secure_filename
import jsonpickle
from omegaconf import OmegaConf
from reasoning.VLMModel import ReasoningModel
import json
import argparse
import logging

logging.basicConfig(
    format="%(message)s", 
    # format="%(asctime)s - %(levelname)s - %(module)s - %(message)s", 
    # datefmt="%m/%d/%Y %I:%M:%S %p",
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("werkzeug").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO) 
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load config
cfg = OmegaConf.load("config.yaml")
cfg = OmegaConf.to_container(cfg, resolve=True)
cfg = OmegaConf.create(cfg)

print("Model: ", cfg.model)
vlm_model = ReasoningModel.create_model(
    cfg.model,
    max_tokens = cfg.max_tokens,
    temperature = cfg.temperature,
    top_p = cfg.top_p,
    reasoning_effort = cfg.reasoning_effort,
    ollama_host = cfg.get('ollama_host', None),
)

app = Flask(__name__)

@app.route('/cmd_vel', methods=['POST'])
def cmd_vel():
    # ... (tu validación inicial de errores se queda igual) ...

    image_file = request.files['image']
    query = request.form['query']
    topology_json = request.form['topology']
    state = request.form['state']    
    previous_movement = request.form.get('prev_movement', None)
    mov_history = request.form.get('mov_history', None)
    
    # --- NUEVOS CAMPOS DE TELEMETRÍA ---
    # Usamos .get() por si usas un cliente de prueba viejo que no los envíe
    telemetry_json_str = request.form.get('telemetry_json', '{}')
    telemetry_text = request.form.get('telemetry_text', '')

    try:
        topology = json.loads(topology_json)
        telemetry_json = json.loads(telemetry_json_str) # Por si el VLM lo necesita como dict
    except Exception as e:
        return jsonify({'error': f'Invalid JSON parsing: {str(e)}'}), 400

    # Convert image
    img_bytes = image_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Call Reasoning model (con manejo de errores del LLM)
    try:
        response = vlm_model.generate_trajectory(
            img, query, 
            topology=topology, 
            state=state,
            previous_movement=previous_movement,
            mov_history=mov_history,
            # --- PASAMOS LA TELEMETRÍA AL MODELO ---
            telemetry_text=telemetry_text, 
        )
    except Exception as e:
        logger.exception("Error calling VLM model: %s", str(e))
        return jsonify({'error': 'Model invocation failed', 'detail': str(e)}), 500
    gpt_response = jsonpickle.encode(response)

    return Response(response=gpt_response, status=200, mimetype="application/json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the reasoning server.')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to run the server on')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the server on')
    args = parser.parse_args()
    
    app.run(debug=True, host=args.host, port=args.port, use_reloader=False)