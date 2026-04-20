#!/usr/bin/env python3
import rospy
import cv2
import json
import requests
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from clover import srv
from std_srvs.srv import Trigger

class CloverVLMClient:
    def __init__(self, server_ip="192.168.1.100", port=5001):
        rospy.init_node('clover_vlm_client')
        
        # Configuración de red para WSL2
        self.url = f"http://{server_ip}:{port}/cmd_vel"
        self.bridge = CvBridge()
        self.last_image = None
        
        # Proxies de servicios de Clover
        rospy.loginfo("Conectando con servicios de Clover...")
        rospy.wait_for_service('get_telemetry')
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        
        # Suscripción a la cámara
        rospy.Subscriber("/main_camera/image_raw", Image, self.image_callback)
        
        rospy.loginfo(f"🚀 Cliente listo. Servidor VLM en: {self.url}")

    def image_callback(self, msg):
        """Almacena el frame más reciente del simulador."""
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error en callback de imagen: {e}")

    def fetch_telemetry_data(self):
        """Obtiene y formatea la telemetría para el servidor Flask."""
        try:
            telem = self.get_telemetry(frame_id='map')
            telemetry_dict = {
                "pose": {"x": telem.x, "y": telem.y, "z": telem.z},
                "orientation": {"yaw": telem.yaw}
            }
            telemetry_text = f"X: {telem.x:.2f}, Y: {telem.y:.2f}, Z: {telem.z:.2f}, Yaw: {telem.yaw:.2f}"
            return telemetry_dict, telemetry_text
        except Exception as e:
            rospy.logwarn(f"Error al obtener telemetría: {e}")
            return {}, "Telemetría no disponible"

    def send_inference_request(self, query, topology, state="Testing"):
        """Envía la petición multipart/form-data al servidor Flask."""
        if self.last_image is None:
            rospy.logwarn("Esperando imagen de la cámara...")
            return None

        # Preparar imagen
        _, img_encoded = cv2.imencode('.jpg', self.last_image)
        files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        # Obtener telemetría real
        telem_json, telem_text = self.fetch_telemetry_data()
        
        # Payload estructurado para server.py
        data = {
            'query': query,
            'topology': json.dumps(topology),
            'state': state,
            'prev_movement': 'None',
            'mov_history': json.dumps([]),
            'telemetry_json': json.dumps(telem_json),
            'telemetry_text': telem_text
        }

        try:
            rospy.loginfo("Enviando petición al VLM...")
            response = requests.post(self.url, files=files, data=data, timeout=35)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            rospy.logerr(f"Fallo en la conexión con el servidor Flask: {e}")
            return None

    def main_loop(self):
        """Bucle de ejecución interactivo incorporando el código de evaluación."""
        # Topología definida para el razonamiento del modelo
        topology = {
            "hallway": ["kitchen", "living room"],
            "kitchen": ["hallway"],
            "living room": ["hallway"]
        }

        while not rospy.is_shutdown():
            print("\n" + "="*50)
            user_query = input("[INPUT] Introduce comando para el dron (o 'exit'): ")
            
            if user_query.lower() in ['exit', 'quit']:
                break

            # Ejecución de la inferencia (Bucle Abierto)
            result = self.send_inference_request(user_query, topology)

            if result:
                # El servidor Flask devuelve el objeto procesado por ReasoningModel
                print("\n🧠 --- RESULTADO DEL RAZONAMIENTO ---")
                print(f"Acción propuesta: {result.get('movement', 'N/A')}")
                print(f"Estado interno:  {result.get('state', 'N/A')}")
                
                # Si el modelo incluye CoT (Chain of Thought)
                if "chain_of_thought" in result:
                    print(f"Razonamiento:\n{result['chain_of_thought']}")
                elif "raw_text" in result:
                    print(f"Respuesta Raw:\n{result['raw_text']}")
                print("="*50)

if __name__ == "__main__":
    # Sustituye con la IP de tu WSL2 (hostname -I)
    client = CloverVLMClient(server_ip="192.168.1.100")
    client.main_loop()