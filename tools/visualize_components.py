import numpy as np
import cv2
import os
from monster.monster import Monster

def Visualize_components():
    files = os.listdir("img")
    monsters = []
    baseline = 256

    for file in files:
        monsters.append(Monster(file, baseline))

    L = len(monsters)

    W = L * baseline
    h = int(baseline * 2.5) + 3 * baseline

    image = np.ones((h, W, 3), dtype=np.uint8) * 255
    for i, monster in enumerate(monsters):
        image[:int(baseline * 2.5), i * baseline:(i + 1) * baseline, :] = monster.picture
        for j, (center, weight) in enumerate(zip(monster.centers, monster.weights)):
            L = center[0]
            text_color = (0, 0, 0) if L > 100 else (255, 255, 255)

            square = np.zeros((baseline, baseline, 3), dtype=np.uint8)
            square[:, :, :] = center
            square = cv2.cvtColor(square, cv2.COLOR_LAB2BGR)

            image[int(baseline * 2.5) + j * baseline:int(baseline * 2.5) + (j + 1) * baseline:,
            i * baseline:(i + 1) * baseline, :] = square

            font_size = 3
            font_thick = 7
            middle_y, middle_x = int(baseline * 2.5) + int((j + .5) * baseline), int((i + .5) * baseline)
            (text_width, text_height), _ = cv2.getTextSize(f"{weight:.2f}",
                                                           cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thick)
            text_x = middle_x - text_width // 2
            text_y = middle_y + text_height // 2
            cv2.putText(image, f"{weight:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, text_color, font_thick)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()