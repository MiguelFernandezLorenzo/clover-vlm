#!/usr/bin/env python3
import rospy
import tf2_ros
import re
import math
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2

class CloverController:
    def __init__(self):
        # Evitamos doble inicialización si el parser ya lo hizo
        if not rospy.core.is_initialized():
            rospy.init_node('clover_controller', anonymous=True)
        
        # Servicios
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.land_srv = rospy.ServiceProxy('land', Trigger)
        
        # Diagnóstico de TF y Nube
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.last_pcl_stamp = None
        self.pcl_sub = rospy.Subscriber('/stereo_camera/points2', PointCloud2, self._pcl_cb)

        # Límites
        self.max_z = 1.5
        self.min_z = 0

    def _pcl_cb(self, msg):
        self.last_pcl_stamp = msg.header.stamp

    def check_vision_health(self):
        """Evaluación crítica de la nube de puntos y las TFs"""
        rospy.loginfo("--- Iniciando Diagnóstico de Visión ---")
        
        # 1. Verificar Nube de Puntos
        if self.last_pcl_stamp is None or (rospy.Time.now() - self.last_pcl_stamp).to_sec() > 1.0:
            rospy.logerr("ERROR: No se recibe Nube de Puntos en /stereo_camera/points2")
        else:
            rospy.loginfo("Nube de puntos: RECIBIDA")

        # 2. Verificar TF de la cámara
        try:
            # Buscamos si existe transformación entre el dron y la cámara óptica
            trans = self.tf_buffer.lookup_transform('base_link', 'left_camera_optical_frame', rospy.Time(0), rospy.Duration(1.0))
            rospy.loginfo(f"TF Cámara: OK (Posición rel: {trans.transform.translation.x:.2f}, {trans.transform.translation.y:.2f})")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"ERROR TF: No se encuentra el frame de la cámara. {e}")
    
    def check_safety(self,z):
        if z > self.max_z or z < self.min_z:
            return False
        return True
        
    def move(self, x, y, z, yaw=0.0, frame='body'):
        """Navegación con control de orientación (Yaw)"""
        z_safe = self.check_safety(z)
        
        # El parámetro 'yaw' define hacia dónde mirará la cámara estéreo
        rospy.loginfo(f"Navegando a X:{x} Y:{y} Z:{z_safe} Yaw:{yaw} rad")
        res = self.navigate(
            x=x, y=y, z=z_safe, 
            yaw=yaw,           # <--- Rotación añadida
            frame_id=frame, 
            speed=0.5, 
            auto_arm=True
        )
        
        if res.success:
            self.wait_for_completion(z_safe)
        return res.success

    def rot(self, angle_deg):
        """Gira el dron un número específico de grados relativo a su frente"""
        yaw_rad = math.radians(angle_deg)
        # x,y,z en 0 en frame 'body' significa 'no te muevas del sitio'
        return self.navigate(x=0, y=0, z=0, yaw=yaw_rad, frame_id='body', speed=0.5)

class CommandParser:
    def __init__(self):
        self.drone = CloverController()
        self.pattern = r"move\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*(?:,\s*(-?\d+\.?\d*))?\s*\)"

    def parse_and_control(self):
        print("\n--- Intérprete Clover Activo ---")
        print("Comandos: 'move(x,y,z)', 'check' (evalúa visión), 'exit'")

        while not rospy.is_shutdown():
            user_input = input(">> ").strip().lower()

            if user_input in ['exit', 'quit']: break
            
            self.oneshot_parse_and_control(user_input)

    def oneshot_parse_and_control(self, user_input):
            
        if 'check' in user_input:
            self.drone.check_vision_health()
            return
        match = re.search(self.pattern, user_input)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            # Si no hay cuarto grupo, yaw = 0
            yaw = float(match.group(4)) if match.group(4) else 0.0
            self.drone.move(x, y, z, yaw=yaw)
        else:
            print("Formato incorrecto. Usa move(x,y,z) o check.")

if __name__ == '__main__':
    try:
        parser = CommandParser()
        parser.parse_and_control()
    except rospy.ROSInterruptException:
        pass