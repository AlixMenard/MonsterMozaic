from utils.distance import components_distance
from monster.monster import Monster
from pathlib import Path
import itertools as it
import numpy as np
import cv2
import os

def Visualize_distances():
    files = os.listdir("img")
    monsters = []
    baseline = 256

    for file in files:
        monsters.append(Monster(file, baseline))

    L = len(monsters)

    distances = np.zeros((L, L))
    for m1, m2 in it.combinations(monsters, 2):
        distances[monsters.index(m1), monsters.index(m2)] = components_distance(m1.centers, m1.weights, m2.centers, m2.weights)
        distances[monsters.index(m2), monsters.index(m1)] = distances[monsters.index(m1), monsters.index(m2)]

    max_dist = np.max(distances)

    w, h = baseline * (L+1), (int(baseline*2.5)+5) * (L+1)
    image = np.ones((h, w, 3), dtype=np.uint8) * 255

    for i, monster in enumerate(monsters):
        y = (int(baseline*2.5)+5) * i
        image[y+2:y+int(baseline*2.5)+2, :baseline, :] = monster.picture
        up = True

        sorted_monsters = np.argsort(distances[i])
        for j in sorted_monsters:
            if i == j: continue
            mw, mh = baseline//2, int(baseline*2.5)//2
            x = calc_x(distances[i, j], max_dist, mw, w)
            if up:
                image[y+2:y+mh+2, x:x+mw, :] = monsters[j].half_picture
                up = False
            else:
                image[y+mh+2:y+2*mh+2, x:x+mw, :] = monsters[j].half_picture
                up = True

    image[(int(baseline*2.5)+5)::(int(baseline*2.5)+5),:,:] = 0

    save_dir = Path("visuals") / "distances"
    num_images = L // 5 + int(L%5!=0)
    for i in range(num_images):
        smaller = image[i*5*(int(baseline*2.5)+5):(i+1)*5*(int(baseline*2.5)+5),:,:]
        cv2.imwrite(str(save_dir / f"distances_{i}.png"), smaller)
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.imshow("image", smaller)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()

def calc_x(dist, max_dist, pic_w, img_w):
    dist_ratio = dist / max_dist
    displacement = int(dist_ratio*(img_w-pic_w))
    return displacement