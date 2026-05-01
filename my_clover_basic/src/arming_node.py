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
        if not rospy.core.is_initialized():
            rospy.init_node('clover_controller', anonymous=True)
        
        rospy.loginfo("Esperando servicios de Clover...")
        rospy.wait_for_service('get_telemetry')
        rospy.wait_for_service('navigate')
        rospy.wait_for_service('land')
        
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.land_srv = rospy.ServiceProxy('land', Trigger)
        
        self.max_z = 2.0
        self.min_z = 0.0

    def wait_for_completion(self, tolerance=0.2, timeout=10.0):
        """
        Espera a que el dron llegue al setpoint. 
        Calcula la distancia Euclidiana usando: 
        d = sqrt(dx^2 + dy^2 + dz^2)
        """
        rospy.loginfo("Verificando movimiento...")
        rospy.sleep(0.5) # <-- VITAL: Damos medio segundo para que Clover publique el nuevo target
        start_time = rospy.get_time()
        
        while not rospy.is_shutdown():
            # 'navigate_target' nos da la distancia relativa al objetivo actual
            telem = self.get_telemetry(frame_id='navigate_target')
            dist = math.sqrt(telem.x**2 + telem.y**2 + telem.z**2)
            
            if dist < tolerance:
                rospy.loginfo("✅ Objetivo alcanzado con éxito.")
                return True
            
            if (rospy.get_time() - start_time) > timeout:
                rospy.logwarn("⚠️ Tiempo de espera agotado (Timeout) - El dron podría estar bloqueado.")
                return False
                
            rospy.sleep(0.2)

    def takeoff(self, alt=1.0):
        """Despegue directo: armar y subir a la vez es lo más seguro en Clover."""
        if alt > self.max_z: alt = self.max_z
        
        rospy.loginfo(f"Iniciando secuencia de despegue a {alt}m...")
        
        # El primer comando DEBE llevar auto_arm=True para que los motores giren
        res = self.navigate(x=0, y=0, z=alt, frame_id='body', auto_arm=True, speed=0.5)
        
        if res.success:
            return self.wait_for_completion()
        else:
            rospy.logerr("❌ El servicio de navegación rechazó el despegue.")
            return False

    def land(self):
        rospy.loginfo("Aterrizando...")
        res = self.land_srv()
        return res.success

    def move(self, x, y, z, yaw=0.0):
        # PROTECCIÓN CONTRA FAILSAFES: Comprobar que el dron está armado/volando
        telem = self.get_telemetry()
        if not telem.armed:
            rospy.logerr("❌ El dron no está armado. ¡Usa 'takeoff(alt)' antes de intentar moverlo!")
            return False

        rospy.loginfo(f"Moviendo a X:{x} Y:{y} Z:{z} Yaw:{yaw}")
        res = self.navigate(x=x, y=y, z=z, yaw=yaw, frame_id='body', speed=0.5)
        if res.success:
            self.wait_for_completion()
        return res.success

    def check_vision_health(self):
        """Método añadido para evitar que 'check' crashee el script."""
        telem = self.get_telemetry()
        rospy.loginfo(f"📊 ESTADO ACTUAL -> Armado: {telem.armed}, Modo: {telem.mode}, Altura: {telem.z:.2f}m")
        return True


class CommandParser:
    def __init__(self):
        self.drone = CloverController()
        self.move_pattern = r"move\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*(?:,\s*(-?\d+\.?\d*))?\s*\)"
        self.takeoff_pattern = r"takeoff\(\s*(\d+\.?\d*)\s*\)"

    def parse_and_control(self):
        print("\n--- Intérprete Clover Activo ---")
        print("Comandos: 'takeoff(z)', 'land()', 'move(x,y,z,yaw)', 'check', 'exit'")

        while not rospy.is_shutdown():
            try:
                user_input = input(">> ").strip().lower()
            except EOFError: break

            if user_input in ['exit', 'quit']: break
            if 'land()' in user_input:
                self.drone.land()
                continue
            if 'check' in user_input:
                self.drone.check_vision_health()
                continue
            
            # Match Takeoff
            tk_match = re.search(self.takeoff_pattern, user_input)
            if tk_match:
                alt = float(tk_match.group(1))
                self.drone.takeoff(alt)
                continue

            # Match Move
            mv_match = re.search(self.move_pattern, user_input)
            if mv_match:
                x = float(mv_match.group(1))
                y = float(mv_match.group(2))
                z = float(mv_match.group(3))
                yaw = float(mv_match.group(4)) if mv_match.group(4) else 0.0
                self.drone.move(x, y, z, yaw=yaw)
                continue

            print("Comando no reconocido.")

if __name__ == '__main__':
    try:
        parser = CommandParser()
        parser.parse_and_control()
    except rospy.ROSInterruptException:
        pass