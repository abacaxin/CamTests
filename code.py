import cv2
import numpy as np

categorias = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
modelo = cv2.dnn.readNetFromONNX('modelo.onnx')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
try: 
    while True:
        ret, frame = cap.read()
        
        if not ret: exit()
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224, 224), swapRB=True, crop=False)
        modelo.setInput(blob)
        saida = modelo.forward()
        indice = np.argmax(saida[0])
    
        cv2.putText(frame, f"{categorias[indice]} ({saida[0][indice]:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Classificação de Resíduos', frame)
    
        if cv2.waitKey(1) == 27:
            break

finally:
    cap.release()   
    cv2.destroyAllWindows()
