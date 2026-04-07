import cv2  # Biblioteca de visão computacional
import numpy as np  # OpenCV usa numpy internamente, bom ter consciência disso

# Abre a webcam. O 0 significa "primeira câmera disponível"
cap = cv2.VideoCapture(0)

# Captura UM frame
ret, frame = cap.read()
# ret = booleano, deu certo?
# frame = a imagem em si

print("Deu certo?", ret)
print("Tipo do objeto:", type(frame))       # numpy.ndarray
print("Shape (formato):", frame.shape)      # (altura, largura, canais de cor)
print("Dtype:", frame.dtype)                # uint8 = valores de 0 a 255

# Mostra a imagem numa janela
cv2.imshow("Meu frame", frame)
cv2.waitKey(0)  # Espera qualquer tecla

# Libera a câmera e fecha janela
cap.release()
cv2.destroyAllWindows()
