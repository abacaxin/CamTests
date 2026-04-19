import cv2
import os

categoria = "plastic"  # muda pra cada material
os.makedirs(f"dataset/{categoria}", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
contador = 0

print(f"Tirando fotos de: {categoria}")
print("Aperta ESPAÇO pra capturar, Q pra sair")

import pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                ret, frame = cap.read()
                nome = f"dataset/{categoria}/{categoria}_{contador}.jpg"
                cv2.imwrite(nome, frame)
                contador += 1
                print(f"Foto {contador} salva!")
            if event.key == pygame.K_q:
                running = False

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

cap.release()
pygame.quit()
