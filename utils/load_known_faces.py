import os
import face_recognition

def load_known_faces(ruta_carpeta = "images"):
    nombres = []
    codificaciones = []
    for nombre in os.listdir(ruta_carpeta):
        try:
            img = face_recognition.load_image_file(os.path.join(ruta_carpeta, nombre))
            # Asegúrate de que se detecte al menos una cara antes de calcular la codificación
            codificaciones_img = face_recognition.face_encodings(img)
            if len(codificaciones_img) > 0:
                codificacion = codificaciones_img[0]
                nombres.append(os.path.splitext(nombre)[0])
                codificaciones.append(codificacion)
            else:
                print(f"No se detectaron caras en la imagen: {nombre}")
        except Exception as e:
            print(f"Error procesando la imagen {nombre}: {e}")
    return nombres, codificaciones