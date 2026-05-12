import os
import json
import logging
import cv2
import numpy as np
from vllm import LLM, SamplingParams
from reasoning.load_prompts import load_system_prompt
from reasoning.VLMModel import VLMBaseModel

logger = logging.getLogger(__name__)

class QwenVLLMDescriptor(VLMBaseModel):
    """
    Descriptor que ejecuta Qwen (3/3.5) de forma nativa usando vLLM.
    No requiere servidor externo; carga el modelo en la VRAM directamente.
    """

    def __init__(self, model_path: str = "~/models/qwen27b_awq", 
                 max_tokens: int = 512,
                 temperature: float = 0.0, 
                 top_p: float = 0.2,
                 gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 4096):
        
        self.max_tokens = max_tokens
        
        # 1. Configurar parámetros de muestreo (Sampling)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<thought>", "</thought>", "```json"]
        )

        # 2. Inicializar el motor vLLM
        # Expandimos la ruta por si usas '~'
        full_model_path = os.path.expanduser(model_path)
        
        # Carga el modelo directamente en el código
        self.engine = LLM(
            model=full_model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            # Si el modelo es AWQ, vLLM lo detecta, pero puedes forzarlo:
            quantization="awq" if "awq" in model_path.lower() else None
        )

    def generate_trajectory(self, image, user_query, topology=None, state=None,
                            telemetry_text="", previous_movement=None, mov_history=None, test=False):
        if test:
            return {"room": "hallway", "movement": "B3", "state": "Test", "description": "vLLM Offline Test"}

        # Preparación de prompts (Igual que en tu código original)
        user_text_parts = [user_query]
        if state: user_text_parts.append(f"Current FSM state: {state}")
        if telemetry_text: user_text_parts.append(f"Telemetry: {telemetry_text}")
        user_text = "\n\n".join(user_text_parts)

        system_prompt = load_system_prompt(
            path_dir="prompts",
            system_prompt_path="system_prompt.txt",
            output_prompt_path="output_prompt.txt",
            curr_state=state
        )

        # vLLM para modelos Vision (Qwen-VL) usa un formato de diccionario para el contenido
        # Nota: Ajustamos al esquema Chat de Qwen 2.5/3 VL
        prompt = f"{system_prompt}\n\n{user_text}"

        # 3. Formatear la entrada para vLLM Vision
        # vLLM espera una imagen como objeto PIL o matriz numpy en un formato específico
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image # vLLM acepta la matriz BGR de OpenCV directamente
            },
        }

        try:
            # 4. Inferencia Síncrona
            outputs = self.engine.generate(inputs, sampling_params=self.sampling_params)
            
            # Extraer texto generado
            generated_text = outputs[0].outputs[0].text
            
            # Procesar JSON (Reutilizando tu lógica)
            from reasoning.parser_helper import parse_json_from_text # Asumiendo que moviste el helper
            json_text = parse_json_from_text(generated_text)
            
            return json.loads(json_text)

        except Exception as e:
            logger.exception("Error en inferencia vLLM: %s", str(e))
            return {"error": str(e)}