import cv2
import numpy as np
import pygame

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Classificacao de Residuos")

modelo = cv2.dnn.readNetFromONNX("modelo.onnx")
categorias = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS, 10)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224,224), swapRB=True)
    modelo.setInput(blob)
    saida = modelo.forward()
    indice = np.argmax(saida)
    confianca = saida[0][indice] * 100

    # Converte frame BGR → RGB pra pygame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640, 480))

    # Escreve texto
    cv2.putText(frame_rgb, f"{categorias[indice]}: {confianca:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostra no pygame
    surface = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

cap.release()
pygame.quit()
