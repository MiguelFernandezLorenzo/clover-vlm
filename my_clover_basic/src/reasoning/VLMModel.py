from abc import ABC, abstractmethod
import base64
import cv2


class VLMBaseModel(ABC):
    @abstractmethod
    def generate_trajectory(self, image: cv2.Mat, 
                            user_query: str, 
                            topology: str, 
                            state: str,
                            previous_movement: str,
                            mov_history: str,
                            test: bool) -> str:
        pass

    def encode_image_base64(self, image):
        """Function to encode the image as base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode("utf-8") 
    
        
class ReasoningModel:
    @staticmethod
    def create_model(model: str, **kwargs) -> VLMBaseModel:
        if "gpt" in model:
            from reasoning.gpt_class import GPTDescriptor
            return GPTDescriptor(model, temperature=kwargs.get("temperature", 0),
                                 max_tokens=kwargs.get("max_tokens", 300),
                                 top_p=kwargs.get("top_p", 0.2), img_type="image/jpeg")
        elif "o4" in model:
            from reasoning.o_models_class import OModelDescriptor
            return OModelDescriptor(model, max_tokens=kwargs.get("max_tokens", 300),
                                    reasoning_effort=kwargs.get("reasoning_effort", "medium"))
        elif "gemini" in model:
            from reasoning.gemini import GeminiDescriptor
            return GeminiDescriptor(model, temperature=kwargs.get("temperature", 0))
        elif "qwen3-vl" in model:
            from reasoning.qwen_ollama import QwenOllamaDescriptor
            return QwenOllamaDescriptor(model,
                                       max_tokens=kwargs.get("max_tokens", 512),
                                       temperature=kwargs.get("temperature", 0.0),
                                       top_p=kwargs.get("top_p", 0.2),
                                       ollama_host=kwargs.get("ollama_host", None))
        elif "qwen3.5" in model:
            from reasoning.qwen_ollama import QwenOllamaDescriptor
            return QwenOllamaDescriptor(model,
                                       max_tokens=kwargs.get("max_tokens", 128),
                                       reasoning_effort=kwargs.get('reasoning_effort', 'low'), # Añadido aquí
                                       temperature=kwargs.get("temperature", 0.0),
                                       top_p=kwargs.get("top_p", 0.2),
                                       ollama_host=kwargs.get("ollama_host", None))
        elif "llama" in model:
            from reasoning.llama_ollama import LlamaOllamaDescriptor
            return LlamaOllamaDescriptor(
                model,
                max_tokens=kwargs.get("max_tokens", 128),
                reasoning_effort=kwargs.get('reasoning_effort', 'low'),
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 0.2),
                ollama_host=kwargs.get("ollama_host", None)
            )
        
        elif "gemma3" in model:
            from reasoning.gemma_ollama import GemmaOllamaDescriptor
            return GemmaOllamaDescriptor(
                model,
                max_tokens=kwargs.get("max_tokens", 256), # Gemma suele manejar contextos bien, subimos un poco
                reasoning_effort=kwargs.get('reasoning_effort', 'low'),
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 0.1), # Un top_p más bajo suele ayudar a la precisión en Gemma
                ollama_host=kwargs.get("ollama_host", None)
            )
        
        else:
            raise ValueError(f"Model not implemented: {model}")