#!/usr/bin/env python3
import rospy
import math
import cv2
import json
import requests
import numpy as np
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from clover import srv 
from std_srvs.srv import Trigger

class CloverVLMClient:
    #192.168.1.128
    def __init__(self, server_ip="127.0.0.1", port=5001):
        rospy.init_node('clover_vlm_client')
        
        self.url = f"http://{server_ip}:{port}/cmd_vel"
        self.bridge = CvBridge()
        self.last_image = None
        
        # World topology for VLM navigation logic
        self.topology = {
            "kitchen": [ "living room"],
            "bedroom": ["kitchen"],
            "living room": ["kitchen", "bedroom"],
        }
        
        rospy.loginfo("Connecting to Clover services...")
        rospy.wait_for_service('get_telemetry')
        rospy.wait_for_service('navigate')
        
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        
        # Map for VLM movement codes (e.g., A3, B1) Using degrees for rotation and meters for translation.
        self.action_map = {
            'A': {'name': 'Forward',  'axis': 'x', 'base': 0.1, 'multiplier': [1,2.5,5]},
            'B': {'name': 'Rotate',     'axis': 'yaw','base': 15,   'multiplier': [1,3,6]},
            'C': {'name': 'Rotate',   'axis': 'yaw', 'base': -15, 'multiplier': [1,3,6]},
            'D': {'name': 'Lateral', 'axis': 'y', 'base': 0.1,  'multiplier': [-1,1]}    
        }

        self.current_nav_goal = None
        self.is_running = True

        # Hilo para mantener vivo el modo OFFBOARD
        self.heartbeat_thread = threading.Thread(target=self.maintain_offboard)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

        rospy.Subscriber("/stereo_camera/right/image_color", Image, self.image_callback)
        rospy.loginfo(f"🚀 Client ready. VLM Server at: {self.url}")

    def image_callback(self, msg):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_image = cv2.resize(self.last_image, (240, 240))
        except Exception as e:
            rospy.logerr(f"Image conversion error: {e}")

    def fetch_telemetry_data(self):
        try:
            telem = self.get_telemetry()
            return {'x': telem.x, 'y': telem.y, 'z': telem.z, 'yaw': telem.yaw}, \
                   f"X: {telem.x:.2f}, Y: {telem.y:.2f}, Z: {telem.z:.2f}, Yaw: {telem.yaw:.2f}"
        except Exception:
            return None, "Telemetry unavailable"

    def maintain_offboard(self):
        rate = rospy.Rate(10) # 10Hz es perfecto para PX4
        while not rospy.is_shutdown() and self.is_running:
            if self.current_nav_goal:
                # Re-enviamos el último comando conocido
                try:
                    self.navigate(**self.current_nav_goal)
                except:
                    pass
            rate.sleep()

    def send_inference_request(self, query, topology, state):
        if self.last_image is None:
            rospy.logwarn("Waiting for camera image...")
            return None
        
        telem_json, telem_text = self.fetch_telemetry_data()
        
        data = {
            'query': query if query else "Find the objective",
            'topology': json.dumps(topology),
            'state': state, # This must match one of the keys in your server's if/elif block
            'telemetry_json': json.dumps(telem_json),
            'telemetry_text': telem_text
        }
        
        _, img_encoded = cv2.imencode('.jpg', self.last_image)
        files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        try:
            response = requests.post(self.url, data=data, files=files, timeout=150)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            rospy.logerr(f"VLM Server communication error: {e}")
            return None
    def perform_panoramic_scan(self):
        rospy.loginfo(" Iniciando escaneo de 360 grados...")
        panoramic_images = []
        
        # 90 grados en radianes
        ninety_degrees = math.radians(90)
        
        for i in range(4):
            rospy.loginfo(f" Giro {i+1}/4...")
            # Giramos 90 grados sobre el eje actual (frame body)
            self.navigate(x=0, y=0, z=0, yaw=ninety_degrees, frame_id='body', relative=True)
            
            # Esperamos a que el giro termine y la imagen se estabilice
            rospy.sleep(2.5) 
            
            if self.last_image is not None:
                # Guardamos una copia de la imagen actual
                panoramic_images.append(self.last_image.copy())
            else:
                rospy.logwarn(" No se pudo capturar imagen en este ángulo.")

        return panoramic_images

    def execute_any_command(self, cmd):
        cmd = cmd.upper().strip()
        
        # 1. Comandos básicos
        if cmd == "TAKEOFF":
            rospy.loginfo("Executing Takeoff...")
            # Despegamos a 1.5m sobre la posición actual
            self.navigate(x=0, y=0, z=1.5, frame_id='body', auto_arm=True)
            rospy.sleep(4)
            
            # Seteamos el objetivo para el heartbeat tras el despegue
            t = self.get_telemetry(frame_id='map')
            self.current_nav_goal = {'x': t.x, 'y': t.y, 'z': t.z, 'yaw': t.yaw, 'frame_id': 'map'}
            return True
        elif cmd == "LAND":
            rospy.loginfo("Executing Landing...")
            # Asegúrate de que el servicio 'land' esté disponible
            try:
                land_srv = rospy.ServiceProxy('land', Trigger)
                land_srv()
                return True
            except rospy.ServiceException as e:
                rospy.logerr(f"Landing failed: {e}")
                return False
        
        # 2. Comandos del Action Map (Ej: A2, B3)
        if len(cmd) >= 2 and cmd[0] in self.action_map:
            letter = cmd[0]
            try:
                # El segundo carácter es el nivel de intensidad (1, 2, 3)
                # Usamos index-1 porque las listas en Python empiezan en 0
                idx = int(cmd[1]) - 1 
                config = self.action_map[letter]
                
                # Validar que el índice exista en los multiplicadores
                if not (0 <= idx < len(config['multiplier'])):
                    rospy.logwarn(f"Índice fuera de rango: {cmd}")
                    return False

                # Cálculo racional: Base * Multiplicador
                # Si es rotación, convertimos a radianes
                base_val = config.get('base', 1.0) # 1.0 por defecto si no hay base (Lateral)
                multiplier = config['multiplier'][idx]
                distance = base_val * multiplier
                
                nav_args = {'x': 0, 'y': 0, 'z': 0, 'yaw': 0, 'frame_id': 'body'}
                
                if config['axis'] == 'yaw':
                    # IMPORTANTE: Convertir grados a radianes para ROS
                    nav_args['yaw'] = math.radians(distance)
                    rospy.loginfo(f"Executing: {config['name']} {distance} degrees")
                else:
                    nav_args[config['axis']] = distance
                    rospy.loginfo(f"Executing: {config['name']} {distance}m")

                self.current_nav_goal = nav_args # Guardamos para el hilo de mantenimiento
                self.navigate(**nav_args)
                rospy.sleep(1)
                return True
                
            except (ValueError, IndexError) as e:
                rospy.logerr(f"Error parsing command {cmd}: {e}")
                return False
        
        return False

    def autonomous_loop(self, initial_query):
        current_state = "Recognize Room" 
        query = initial_query
        
        # --- VERIFICACIÓN DE SEGURIDAD ANTES DE DESPEGAR ---
        telem_start = self.get_telemetry()
        
        # Si no está armado, o si está armado pero muy cerca del suelo (z < 0.3m)
        if not telem_start.armed or telem_start.z < 0.3:
            rospy.loginfo("Dron en tierra o desarmado. Iniciando TAKEOFF...")
            self.execute_any_command("TAKEOFF")
        else:
            rospy.loginfo(f"Dron ya en vuelo (Altitud: {telem_start.z:.2f}m). Saltando TAKEOFF.")
        # --------------------------------------------------

        while not rospy.is_shutdown():
            # 1. OBTENER POSICIÓN ACTUAL ANTES DE LA ESPERA
            telem = self.get_telemetry(frame_id='map')
            self.current_nav_goal = {
                'x': telem.x, 'y': telem.y, 'z': telem.z, 
                'yaw': telem.yaw, 'frame_id': 'map'
            }
            # 2. ENVIAR COMANDO DE "HOLD" (Mantener posición actual)
            # Esto refresca el modo OFFBOARD justo antes de que el script se bloquee con el VLM
            self.navigate(x=telem.x, y=telem.y, z=telem.z, yaw=telem.yaw, frame_id='map')
            
            rospy.loginfo(f"🔄 Waiting for VLM response in State: {current_state}...")
            
            # La ejecución se bloquea aquí. Si COM_OBL_ACT es 0, el dron se quedará quieto.
            result = self.send_inference_request(query, self.topology, current_state)
            
            if result and 'response' in result:
                try:
                    res_data = json.loads(result.get('response', '{}'))
                    cmd = res_data.get('movement', 'WAIT')
                    new_state = res_data.get('state', current_state)
                    
                    rospy.loginfo(f"🤖 VLM Response: {cmd} | Next State: {new_state}")

                    if cmd != 'WAIT':
                        self.execute_any_command(cmd)
                    
                    current_state = new_state

                    if cmd == "LAND":
                        self.execute_any_command("LAND")
                        break
                except Exception as e:
                    rospy.logerr(f"JSON Error: {e}")
            
            rospy.sleep(1)
if __name__ == "__main__":
    client = CloverVLMClient()
    # You can change this query based on the task
    client.autonomous_loop("Navigate to the kitchen and find the bottle")