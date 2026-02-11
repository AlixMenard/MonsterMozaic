import numpy as np
import cv2

def lab_to_rgb_tuple(lab_color):
    lab_pixel = np.uint8([[lab_color]])
    bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
    rgb = bgr_pixel[0][0][::-1]
    return tuple(rgb)


def print_colored_text(text, lab_color):
    r, g, b = lab_to_rgb_tuple(lab_color)
    print(f"\033[38;2;{r};{g};{b}m{text}\033[0m")