#!/usr/bin/env python3
import rospy
import cv2
import json
import requests
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from clover import srv  # Importante: asegurarnos de tener los servicios
from std_srvs.srv import Trigger
import math

class CloverVLMClient:
    def __init__(self, server_ip="192.168.1.100", port=5001):
        rospy.init_node('clover_vlm_client')
        
        self.url = f"http://{server_ip}:{port}/cmd_vel"
        self.bridge = CvBridge()
        self.last_image = None
        
        rospy.loginfo("Conectando con servicios de Clover...")
        rospy.wait_for_service('get_telemetry')
        rospy.wait_for_service('navigate') # <-- NUEVO: Esperar al servicio navigate
        
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate) # <-- NUEVO: Proxy para mover el dron
        
        self.action_map = {
            'A': {'name': 'Avanzar', 'axis': 'x', 'multiplier': 0.1}, # A1=0.1m, A3=1m
            'B': {'name': 'Lateral', 'axis': 'y', 'multiplier': 0.1},  # B1=0.1m, B3=1m
            'C': {'name': 'Rotar',   'axis': 'yaw', 'multiplier': 0.2}, # C en radianes
            'D': {'name': 'Vertical', 'axis': 'z', 'multiplier': 0.2}
        }

        rospy.Subscriber("/stereo_camera/right/image_color", Image, self.image_callback)
        rospy.loginfo(f"🚀 Cliente listo. Servidor VLM en: {self.url}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error al convertir imagen: {e}")
            self.last_image = None
    
    def fetch_telemetry_data(self):
        try:
            telemetry = self.get_telemetry()
            return {
                'x': telemetry.x,
                'y': telemetry.y,
                'z': telemetry.z,
                'yaw': telemetry.yaw
            }
        except Exception as e:
            rospy.logerr(f"Error al obtener telemetría: {e}")
            return None
    def send_inference_request(self, query, topology):
        if self.last_image is None:
            rospy.logwarn("No se ha recibido ninguna imagen aún. Esperando...")
            return None
        
        telemetry = self.fetch_telemetry_data()
        if telemetry is None:
            rospy.logwarn("No se pudo obtener telemetría. Intentando de nuevo...")
            return None
        
        payload = {
            'query': query,
            'telemetry': telemetry,
            'topology': topology
        }
        
        _, img_encoded = cv2.imencode('.jpg', self.last_image)
        files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        try:
            response = requests.post(self.url, data={'payload': json.dumps(payload)}, files=files, timeout=150)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Error al comunicarse con el servidor VLM: {e}")
            return None
    
    def parse_and_execute(self, command_code):
        """
        Interpreta códigos como 'A3' o 'B1'
        """
        if len(command_code) < 2:
            rospy.logwarn(f"Código inválido: {command_code}")
            return False

        letra = command_code[0].upper()
        try:
            grado = int(command_code[1:])
        except ValueError:
            rospy.logerr("Grado de acción no es un número")
            return False

        if letra in self.action_map:
            config = self.action_map[letra]
            distancia = grado * config['multiplier']
            
            # Construir argumentos para navigate
            nav_args = {'x': 0, 'y': 0, 'z': 0, 'yaw': 0, 'frame_id': 'body'}
            nav_args[config['axis']] = distancia
            
            rospy.loginfo(f"Ejecutando: {config['name']} {distancia}m")
            self.navigate(**nav_args)
            rospy.sleep(2) # Tiempo de estabilización
            return True
        
        elif letra == 'L': # Supongamos L para Land
            self.execute_vlm_command("LAND")
            return True
            
        return False

    def autonomous_loop(self, initial_query):
        current_state = "Recognize Room" # Estado inicial
        query = initial_query
        
        while not rospy.is_shutdown():
            rospy.loginfo(f"🔄 Ciclo Autónomo - Estado actual: {current_state}")
            
            # 1. Pedir razonamiento al VLM
            result = self.send_inference_request(query, self.topology, current_state)
            
            if not result:
                rospy.logerr("Fallo en comunicación. Reintentando...")
                rospy.sleep(2)
                continue

            # 2. Extraer respuesta (asumiendo que el VLM devuelve JSON con 'movement' y 'next_state')
            try:
                # Si el VLM devuelve el JSON directamente en el campo 'response'
                res_data = json.loads(result.get('response', '{}'))
                cmd = res_data.get('movement', 'WAIT')
                current_state = res_data.get('state', current_state) # El VLM decide el siguiente estado
                
                rospy.loginfo(f"🤖 VLM dice: Accion={cmd}, Nuevo Estado={current_state}")

                # 3. Ejecutar acción si no es espera
                if cmd != 'WAIT':
                    success = self.parse_and_execute(cmd)
                    if not success:
                        rospy.logwarn("No se pudo ejecutar el comando.")
                
                # 4. Condición de parada (ejemplo)
                if current_state == "Goal Reached" or cmd == "LAND":
                    rospy.loginfo("Objetivo alcanzado. Aterrizando.")
                    break

            except Exception as e:
                rospy.logerr(f"Error procesando respuesta: {e}")
                
            rospy.sleep(1) # Pequeña pausa entre iteraciones de pensamiento

    def execute_vlm_command(self, movement_cmd):
        """Traduce el comando del VLM a instrucciones reales de Clover"""
        rospy.loginfo(f"Ejecutando comando: {movement_cmd}")
        
        # Ejemplo de traducción de comandos. ¡Ajusta esto a lo que devuelva tu VLM!
        if movement_cmd == "TAKEOFF":
            # auto_arm=True soluciona el error de "Copter is not in OFFBOARD mode"
            self.navigate(x=0, y=0, z=1.5, frame_id='body', auto_arm=True)
            rospy.sleep(3) # Esperar a que despegue
            
        elif movement_cmd == "FORWARD":
            self.navigate(x=1.0, y=0, z=0, frame_id='body', auto_arm=False)
            
        elif movement_cmd == "LEFT":
            self.navigate(x=0, y=1.0, z=0, frame_id='body', auto_arm=False)
            
        elif movement_cmd == "LAND":
            rospy.wait_for_service('land')
            land = rospy.ServiceProxy('land', Trigger)
            land()
        else:
            rospy.logwarn(f"Comando desconocido o no ejecutable: {movement_cmd}. Manteniendo posición.")

    # ... [Mantén tu método send_inference_request igual] ...

    def main_loop(self):
        topology = {
            "hallway": ["kitchen", "living room"],
            "kitchen": ["hallway"],
            "living room": ["hallway"]
        }

        # Opcional: Despegue automático al iniciar para evitar problemas de offboard
        # print("Despegando dron para iniciar operaciones...")
        # self.execute_vlm_command("TAKEOFF")

        while not rospy.is_shutdown():
            print("\n" + "="*50)
            user_query = input("[INPUT] Introduce comando para el dron (o 'exit'): ")
            
            if user_query.lower() in ['exit', 'quit']:
                break

            result = self.send_inference_request(user_query, topology)
            raw_response = result.get('response','N/A') if result else 'N/A'
            
            try:
                response = json.loads(raw_response) if raw_response != 'N/A' else {}
            except json.JSONDecodeError:
                response = {}
                rospy.logerr("Error decodificando el JSON del VLM.")

            if result:
                movimiento = response.get('movement', 'N/A')
                
                print("\n🧠 --- RESULTADO DEL RAZONAMIENTO ---")
                print(f"Acción propuesta: {movimiento}")
                print(f"Estado interno:  {response.get('state', 'N/A')}")
                print("="*50)
                
                # --- NUEVO: Ejecutar realmente el comando en el simulador ---
                if movimiento != 'N/A':
                    self.execute_vlm_command(movimiento)

if __name__ == "__main__":
    client = CloverVLMClient(server_ip="192.168.1.128")
    client.main_loop()