import os
import face_recognition

def load_known_faces(ruta_carpeta="images"):
    nombres = []
    codificaciones = []
    for subcarpeta in os.listdir(ruta_carpeta):
        subcarpeta_path = os.path.join(ruta_carpeta, subcarpeta)
        if os.path.isdir(subcarpeta_path):  # Verificar si es una subcarpeta
            for nombre in os.listdir(subcarpeta_path):
                try:
                    img_path = os.path.join(subcarpeta_path, nombre)
                    img = face_recognition.load_image_file(img_path)
                    # Asegúrate de que se detecte al menos una cara antes de calcular la codificación
                    codificaciones_img = face_recognition.face_encodings(img)
                    if len(codificaciones_img) > 0:
                        codificacion = codificaciones_img[0]
                        nombres.append(subcarpeta)  # Usar el nombre de la subcarpeta
                        codificaciones.append(codificacion)
                    else:
                        print(f"No se detectaron caras en la imagen: {img_path}")
                except Exception as e:
                    print(f"Error procesando la imagen {img_path}: {e}")
    return nombres, codificaciones