#!/usr/bin/env python3
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageManager:
    def __init__(self):
        rospy.init_node('clover_image_manager', anonymous=True)
        self.bridge = CvBridge()
        
        # Variable para almacenar los datos RAW (numpy array)
        self.last_snapshot = None
        
        # Suscriptor: Actualiza la variable constantemente para tener siempre el frame más reciente
        self.image_sub = rospy.Subscriber("/main_camera/image_raw", Image, self._update_image_cb)
        
        # Timer: Ejecuta una acción cada 5 segundos
        self.timer = rospy.Timer(rospy.Duration(5.0), self._periodic_task)
        
        rospy.loginfo("Gestor de imágenes iniciado. Timer cada 5s activo.")

    def _update_image_cb(self, data):
        """Actualiza la variable local con los píxeles de la imagen (RAW)"""
        try:
            # Convertimos a formato OpenCV (BGR8) y guardamos en la variable de clase
            self.last_snapshot = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error en update_image: {e}")

    def _periodic_task(self, event):
        """Tarea que se ejecuta automáticamente cada 5 segundos"""
        if self.last_snapshot is not None:
            rospy.loginfo("Evento 5s: Imagen actual lista en memoria.")
            # Aquí podrías añadir lógica de análisis automático si lo deseas
        else:
            rospy.logwarn("Evento 5s: Aún no se ha recibido ninguna imagen.")

    def save_current_image(self, filename="manual_capture.jpg"):
        """Método para guardar el 'last_snapshot' en un fichero por petición explícita"""
        if self.last_snapshot is not None:
            # Clonamos la imagen para evitar problemas de escritura si el callback intenta actualizarla
            image_to_save = self.last_snapshot.copy()
            cv2.imwrite(filename, image_to_save)
            rospy.loginfo(f"Imagen guardada explícitamente como: {filename}")
            return True
        else:
            rospy.logwarn("Petición de guardado fallida: No hay imagen en memoria.")
            return False

if __name__ == '__main__':
    manager = ImageManager()
    
    # Ejemplo de cómo llamar al método de guardado por petición explícita
    # En un escenario real, esto podría ser activado por un servicio o una condición lógica
    try:
        while not rospy.is_shutdown():
            # Simulamos una petición explícita después de 12 segundos
            rospy.sleep(12)
            manager.save_current_image(f"snapshot_{int(rospy.get_time())}.jpg")
    except rospy.ROSInterruptException:
        pass
