import cv2
import face_recognition
import numpy as np
from utils.load_known_faces import load_known_faces

# Cargar rostros conocidos
detected_names, detected_faces = load_known_faces()

# Iniciar la captura de video
video_capture = cv2.VideoCapture(0)

process_this_frame = True
face_locations = []
recognized_faces = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convertir el frame a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Actualizar la posición de las caras en cada frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Realizar reconocimiento facial cada otro frame
    if process_this_frame:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        recognized_faces = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(detected_faces, face_encoding)
            name = "Desconocido"

            if True in matches:
                first_match_index = matches.index(True)
                name = detected_names[first_match_index]

            recognized_faces.append(name)

    # Dibujar los rectángulos y nombres en cada frame
    for (top, right, bottom, left), name in zip(face_locations, recognized_faces):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Alternar el booleano para procesar cada otro frame
    process_this_frame = not process_this_frame

    # Mostrar el frame en la ventana
    window_name = 'Reconocimiento Facial'
    cv2.imshow(window_name, frame)

    # Verificar si la ventana fue cerrada o si se presionó 'q'
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()