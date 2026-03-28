import requests
import cv2
from io import BytesIO
import json
import rospy

class VLMTestClient:
    def __init__(self, server_url="http://127.0.0.1:5001/cmd_vel"):
        # Inicializamos un nuevo nodo independiente
        rospy.init_node('vlm_http_client', anonymous=True)
        self.server_url = server_url
        self.bridge = cv2.CvBridge()
        self.last_image = None
        
        # 1. Suscripción a la cámara
        rospy.Subscriber("/main_camera/image_raw", Image, self.image_callback)
        
        # 2. Conexión a la telemetría de Clover
        rospy.loginfo("Esperando servicio de telemetría de Clover...")
        rospy.wait_for_service('get_telemetry')
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        
        rospy.loginfo("✅ Cliente de prueba VLM listo. Conectando a: " + self.server_url)

    def image_callback(self, data):
        """Mantiene siempre el frame más reciente en memoria."""
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except cv2.CvBridgeError as e:
            rospy.logerr(f"Error procesando imagen: {e}")

    def get_telemetry_text(self):
        """Extrae la telemetría y la formatea como texto para el LLM."""
        try:
            telem = self.get_telemetry(frame_id='map')
            return f"X: {telem.x:.2f}, Y: {telem.y:.2f}, Z: {telem.z:.2f}, Yaw: {telem.yaw:.2f} rad"
        except Exception as e:
            rospy.logwarn(f"No se pudo obtener telemetría: {e}")
            return "Telemetría no disponible"

    def request_inference(self, query, topology):
        """Empaqueta la imagen, telemetría y query, y hace la petición HTTP."""
        if self.last_image is None:
            rospy.logwarn("Aún no hay imágenes de la cámara. Revisa Gazebo.")
            return None

        # Congelamos el frame actual
        current_img = self.last_image.copy()
        _, img_encoded = cv2.imencode('.jpg', current_img)
        files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        telemetry_text = self.get_telemetry_text()
        
        data = {
            'query': query,
            'topology': json.dumps(topology),
            'state': 'Testing Open Loop',
            'prev_movement': 'None',
            'mov_history': '[]',
            'telemetry_text': telemetry_text
        }

        rospy.loginfo("Enviando datos al servidor VLM... Esperando respuesta 🧠")
        try:
            # Timeout alto porque los LLMs pueden tardar varios segundos en procesar imágenes
            response = requests.post(self.server_url, files=files, data=data, timeout=30.0)
            response.raise_for_status() # Lanza error si el servidor devuelve 404, 500, etc.
            return response.json()
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Error de conexión con el servidor Flask: {e}")
            return None

    def run_interactive_test(self):
        """Bucle principal para interactuar por consola."""
        # Topología de prueba (puedes cambiarla según tu mundo en Gazebo)
        topology = {
            "living room": ["bedroom", "hallway"], 
            "bedroom": ["living room"]
        }

        while not rospy.is_shutdown():
            query = input("\n[USER] Ingresa el comando (ej: 'Ve a la cocina') o 'exit': ")
            if query.lower() in ['exit', 'quit']:
                rospy.loginfo("Cerrando cliente de prueba.")
                break

            # Llamamos al servidor
            result = self.request_inference(query, topology)
            
            # Mostramos los resultados parseados
            if result:
                print("\n" + "="*60)
                print(f"🎯 QUERY: {query}")
                print("-" * 60)
                
                # Extraemos el Chain of Thought (Razonamiento)
                cot = result.get("chain_of_thought", "El servidor no devolvió razonamiento.")
                print(f"🤔 CADENA DE PENSAMIENTO (CoT):\n{cot}")
                
                print("-" * 60)
                # Extraemos la acción inferida
                mov = result.get("movement", "None")
                state = result.get("state", "Unknown")
                print(f"🚀 ACCIÓN DECIDIDA: {mov}")
                print(f"📍 ESTADO INTERNO:  {state}")
                print("="*60 + "\n")

def get_cmd_vel(img, query, topology, state, addr="http://127.0.0.1:5001"):
    url = addr + '/cmd_vel'

    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = BytesIO(img_encoded.tobytes())

    # Simulamos la telemetría que generaría la clase TelemetryPreprocessor
    mock_telemetry_dict = {
        "pose": {"x_meters": 1.5, "y_meters": -2.0, "altitude_meters": 1.1},
        "orientation": {"yaw_degrees": 90.0}
    }
    mock_telemetry_text = "El dron se encuentra en las coordenadas globales X:1.5, Y:-2.0 a una altura de 1.1m. Su orientación actual es de 90.0 grados."

    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'query': query,
        'topology': json.dumps(topology),
        'state': state,
        'prev_movement': 'A1',
        'mov_history': '["Start", "A1"]',
        # --- ENVIAMOS LA TELEMETRÍA ---
        'telemetry_json': json.dumps(mock_telemetry_dict),
        'telemetry_text': mock_telemetry_text
    }

    response = requests.post(url, files=files, data=data)

    try:
        print(response.json())
    except Exception as e:
        print("Error decoding response:", e)
        print(response.text)

    return response.json()

def run_open_loop_test(self, topology):
        rospy.loginfo("--- Iniciando Modo de Evaluación VLM (Bucle Abierto) ---")
        rospy.loginfo("El dron NO se moverá. Solo evaluaremos la inferencia.")
        
        # Opcional: Despegar y quedarse flotando para tener buena vista, 
        # o puedes hacerlo desde el suelo si la cámara ve bien.
        # self.drone.move_relative(0, 0, 1.0) 
        # rospy.sleep(3)

        while not rospy.is_shutdown():
            user_input = input("\n[EVALUACIÓN] Introduce query (ej: 'Ve a la cocina') o 'exit': ")
            
            if user_input.lower() in ['exit', 'quit']:
                break

            state = "Testing Inference"
            prev_movement = "None"
            mov_history = []
            
            # 1. Percepción: Capturamos el estado actual
            img = self.vision.get_current_image()
            pos, yaw = self.drone.get_state()
            
            if img is None:
                rospy.logwarn("Aún no hay imagen de la cámara. Espera un momento...")
                continue
                
            rospy.loginfo("Enviando telemetría e imagen al servidor VLM...")
            
            # 2. Cognición: Petición HTTP a tu servidor Flask
            # (Asegúrate de tener un timeout alto, ej. 15-30s, por si el VLM tarda)
            response = self.request_action(
                img=img, 
                query=user_input, 
                topology=topology, 
                state=state, 
                prev_movement=prev_movement, 
                mov_history=mov_history
            )
            
            # 3. LOGGING: Ver la inferencia sin actuar
            rospy.loginfo("\n" + "="*40)
            rospy.loginfo("🧠 RESPUESTA DEL VLM:")
            rospy.loginfo(f"Query enviado: {user_input}")
            rospy.loginfo(f"Comando inferido (Parseado): {response.get('movement')}")
            rospy.loginfo(f"Estado interno inferido: {response.get('state')}")
            
            # Si tu servidor Flask devuelve el texto RAW o justificación del LLM, imprímelo aquí:
            if "raw_text" in response:
                rospy.loginfo(f"Justificación del VLM:\n{response.get('raw_text')}")
            
            rospy.loginfo("="*40 + "\n")
            
            # ¡No hay código de movimiento aquí! Bucle abierto completado.

if __name__ == "__main__":
    img = cv2.imread('test.png')
    
    query = 'Describe what you perceive and how you reason to achieve that thought'
    # Topology (example)
    topology = {
        "hallway": ["kitchen", "living room"],
        "kitchen": ["hallway"],
        "living room": ["hallway"]
    }
    state = 'Test'
    
    if img is None:
        print("Image not found.")
    else:
        get_cmd_vel(img, query, topology, state)