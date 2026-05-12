from abc import ABC, abstractmethod
import cv2

class VLMBaseModel(ABC):
    @abstractmethod
    def generate_trajectory(self, image: cv2.Mat, 
                            user_query: str, 
                            topology: str, 
                            state: str,
                            previous_movement: str,
                            mov_history: str,
                            test: bool) -> dict: # Retornamos dict para consistencia JSON
        pass


    
        
class ReasoningModel:
    @staticmethod
    def create_model(model: str, **kwargs) -> VLMBaseModel:
        """
        Factory exclusivo para la rama vLLM. 
        Maneja la carga local de pesos para arquitecturas Qwen, Llama y Gemma.
        """
        # Configuración común para vLLM
        vllm_config = {
            "model_path": kwargs.get("model_path", f"~/models/{model}"),
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 0.2),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.85),
            "max_model_len": kwargs.get("max_model_len", 4096)
        }

        if "qwen" in model.lower():
            from reasoning.qwen_vllm import QwenVLLMDescriptor
            return QwenVLLMDescriptor(**vllm_config)

        elif "llama" in model.lower():
            from reasoning.llama_vllm import LlamaVLLMDescriptor
            return LlamaVLLMDescriptor(**vllm_config)

        elif "gemma" in model.lower():
            from reasoning.gemma_vllm import GemmavLLMDescriptor
            # Ajuste específico para Gemma (top_p más bajo por estabilidad)
            vllm_config["top_p"] = kwargs.get("top_p", 0.1)
            return GemmavLLMDescriptor(**vllm_config)

        else:
            raise ValueError(f"Motor vLLM no implementado para el modelo: {model}. "
                             f"Esta rama solo soporta ejecuciones nativas locales.")