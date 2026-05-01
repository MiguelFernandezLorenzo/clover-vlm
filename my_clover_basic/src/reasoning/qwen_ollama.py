import os
import json
import base64
import logging
import requests
import cv2
import numpy as np
import time
from dotenv import load_dotenv
from reasoning.load_prompts import load_system_prompt
from reasoning.VLMModel import VLMBaseModel

load_dotenv()
logger = logging.getLogger(__name__)



def parse_json_from_text(text: str):
    # Similar helper to extract JSON fenced in ```json blocks
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_text = "\n".join(lines[i+1:])
            json_text = json_text.split("```")[0]
            return json_text
    # fallback: try to find first { and last }
    try:
        start = text.index('{')
        end = text.rindex('}')
        return text[start:end+1]
    except Exception:
        return text


class QwenOllamaDescriptor(VLMBaseModel):
    """Descriptor that calls Qwen-VL hosted via Ollama (local or cloud).

    Expects environment variables `OLLAMA_HOST` (optional, default http://localhost:11434)
    and `OLLAMA_API_KEY` (optional, for cloud).
    """

    def __init__(self, model: str = "qwen3-vl:8b", max_tokens: int = 512,
                 temperature: float = 0.0, top_p: float = 0.2,
                 img_type: str = "image/jpeg", ollama_host: str = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.img_type = img_type
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.api_key = os.getenv("OLLAMA_API_KEY")

    def _encode_image_b64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_trajectory(self, image, user_query, topology=None, state=None,
                            telemetry_text="", previous_movement=None, mov_history=None, test=False):
        if test:
            return {
                "room": "hallway",
                "movement": "B3",
                "state": "Oriented Towards Door",
                "description": "Visible hallway with a door, move robot to face the door"
            }

        user_text_parts = [user_query]
        if state:
            user_text_parts.append("Current FSM state: " + state)
        if topology:
            user_text_parts.append("Topology: " + json.dumps(topology))
        if previous_movement:
            user_text_parts.append("Previous movement command: " + previous_movement)
        if telemetry_text:
            user_text_parts.append("Telemetry: " + telemetry_text)

        user_text = "\n\n".join(user_text_parts)

        system_prompt = load_system_prompt(
            path_dir="prompts",
            system_prompt_path="system_prompt.txt",
            output_prompt_path="output_prompt.txt",
            curr_state=state
        )

        # 2. El prompt final DEBE ser 100% texto humano legible
        prompt_limpio = system_prompt + "\n\n" + user_text

        # 3. Codificar la imagen a Base64 puro (sin prefijos "data:image/jpeg;base64,")
        img_b64 = self._encode_image_b64(image)

        # 4. Construir el payload separando "prompt" e "images"
        payload = {
            "model": self.model,
            "prompt": prompt_limpio,     # <-- Solo el texto de tus prompts
            "images": [img_b64],         # <-- La imagen aislada en su propia lista
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        url = f"{self.ollama_host.rstrip('/')}/api/generate"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            text = resp.text
            # Try to extract JSON from response
            json_text = parse_json_from_text(text)
            try:
                return json.loads(json_text)
            except Exception:
                # As fallback, return raw text
                logger.debug("Could not parse JSON from Ollama response, returning raw text")
                return {"raw": text}

        except Exception as e:
            logger.exception("Error calling Ollama Qwen-VL: %s", str(e))
            return {"error": str(e)}
