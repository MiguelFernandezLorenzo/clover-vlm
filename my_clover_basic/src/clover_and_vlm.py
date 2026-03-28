#!/usr/bin/env python3
import rospy
import actionlib
import re
from my_clover_msgs.msg import VLMInferenceAction, VLMInferenceGoal
from clover import srv
from std_srvs.srv import Trigger

class CloverVLPPipeline:
    def __init__(self):
        rospy.init_node('clover_vlm_pipeline')

        # 1. Conexión con el ActionServer del VLM
        self.vlm_client = actionlib.SimpleActionClient('vlm_inference', VLMInferenceAction)
        rospy.loginfo("Esperando al servidor vlm_action_srv...")
        self.vlm_client.wait_for_server()

        # 2. Conexión con los servicios de Clover (Simulación)
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.land = rospy.ServiceProxy('land', Trigger)

        # 3. Regex para decodificar la respuesta de Ollama
        self.move_pattern = r"move\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)"

        rospy.loginfo("Pipeline de Clover con VLM listo.")

    def execute_move(self, x, y, z):
        """Envía el comando de movimiento al Clover."""
        rospy.loginfo(f"🚀 Navegando a: x={x}, y={y}, z={z}")
        # Usamos frame_id='body' para que el movimiento sea relativo al dron
        # o 'map' si el VLM da coordenadas absolutas de la small_house
        self.navigate(x=x, y=y, z=z, frame_id='body', speed=0.5, auto_arm=True)

    def run(self):
        while not rospy.is_shutdown():
            user_input = input("\n[USER] Introduce comando (ej: 'Busca la mesa'): ")
            
            if user_input.lower() in ['exit', 'land']:
                self.land()
                break

            # A. Enviar meta al ActionServer (Inteligencia)
            goal = VLMInferenceGoal(user_command=user_input)
            self.vlm_client.send_goal(goal)
            
            rospy.loginfo("Pensando con Ollama (Llama 3.2 Vision)...")
            self.vlm_client.wait_for_result()
            
            result = self.vlm_client.get_result()
            vlm_text = result.inferred_action
            
            rospy.loginfo(f"🤖 VLM Sugiere: {vlm_text}")

            # B. Decodificar la respuesta (Cohesión)
            match = re.search(self.move_pattern, vlm_text, re.IGNORECASE)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                
                # C. Ejecución (Navegación)
                self.execute_move(x, y, z)
            else:
                rospy.logwarn("⚠️ El VLM no devolvió un comando move(x,y,z) válido.")

if __name__ == '__main__':
    try:
        pipeline = CloverVLPPipeline()
        pipeline.run()
    except rospy.ROSInterruptException:
        pass
