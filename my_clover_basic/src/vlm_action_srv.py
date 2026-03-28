import rospy
import actionlib
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from my_clover_msgs.msg import VLMInferenceAction, VLMInferenceResult, VLMInferenceFeedback
from llama_api_client import LlamaVLMClient

class VLMActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('vlm_inference', VLMInferenceAction, self.execute, False)
        self.bridge = CvBridge()
        self.client_ollama = LlamaVLMClient()
        self.last_image = None
        
        # Suscripción a la cámara de Gazebo
        # TODO: cambiar a camara/s stereo
        # TODO: 
        rospy.Subscriber("/main_camera/image_raw", Image, self.image_callback)
        
        self.server.start()
        rospy.loginfo("VLM Action Server Iniciado")

    def image_callback(self, msg):
        # Mantenemos siempre el último frame disponible en memoria
        self.last_image = msg

    def execute(self, goal):
        feedback = VLMInferenceFeedback()
        result = VLMInferenceResult()

        # 1. Validación de racionalidad: ¿Hay imagen?
        if self.last_image is None:
            self.server.set_aborted(result, "No hay frames de cámara disponibles")
            return

        feedback.status = "Capturando imagen y procesando..."
        self.server.publish_feedback(feedback)

        # 2. Guardar imagen temporalmente
        cv_img = self.bridge.imgmsg_to_cv2(self.last_image, "bgr8")
        img_path = "/tmp/vlm_capture.jpg"
        cv2.imwrite(img_path, cv_img)

        # 3. Consulta al VLM (Ollama)
        # Aquí inyectas tus prompts de política que ya tienes preparados
        policy = "Tu política de decisión aquí..."
        
        rospy.loginfo(f"Consultando VLM para: {goal.user_command}")
        response = self.client_ollama.query(policy, goal.user_command, img_path)

        # 4. Enviar resultado
        result.inferred_action = response
        result.success = True
        self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('vlm_action_server')
    VLMActionServer()
    rospy.spin()
