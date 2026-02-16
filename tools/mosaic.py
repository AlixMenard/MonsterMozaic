from utils.distance import components_distance
from monster.monster import Monster
import itertools as it
import numpy as np
import cv2
import os


def cost(array, distances):
    w, h = array.shape
    total_cost = 0
    for i in range(w):
        for j in range(h):
            idx1 = array[i, j]
            if idx1 == -1: continue
            for k in range(w):
                for l in range(h):
                    idx2 = array[k, l]
                    if (i, j) == (k, l) or idx2 == -1: continue
                    grid_dist = abs(i - k) + abs(j - l)
                    total_cost -= distances[idx1, idx2] / grid_dist
    return total_cost


def get_local_cost(r, c, array, distances):
    w, h = array.shape
    local_energy = 0
    idx1 = array[r, c]
    if idx1 == -1: return 0

    for i in range(w):
        for j in range(h):
            idx2 = array[i, j]
            if (i, j) == (r, c) or idx2 == -1: continue
            grid_dist = np.sqrt((r - i) ** 2 + (c - j) ** 2)
            local_energy -= distances[idx1, idx2] / grid_dist
    return local_energy


def mosaic(compo: dict, num_swaps: int = 10**5):
    list_monsters = os.listdir("img")
    monsters = [Monster(file) for file in list_monsters]
    monster_names = [monster.name for monster in monsters]

    L = len(monsters)
    distances = np.zeros((L, L))
    for i, j in it.combinations(range(L), 2):
        m1, m2 = monsters[i], monsters[j]
        dist = components_distance(m1.centers, m1.weights, m2.centers, m2.weights)
        distances[i, j] = distances[j, i] = dist

    max_d = distances.max()
    if max_d > 0: distances /= max_d

    name_to_idx = {name: i for i, name in enumerate(monster_names)}
    roaster = []
    for name, count in compo.items():
        idx = name_to_idx[name]
        roaster.extend([idx] * count)
    np.random.shuffle(roaster)

    width = (len(roaster) + 1) // 2
    m = np.full((2, width), -1, dtype=int)
    for i, monster_idx in enumerate(roaster):
        m[i % 2, i // 2] = monster_idx

    curr_c = cost(m, distances)
    # print(f"Initial cost: {curr_c}")

    T = 1.0  # Temperature
    cooling = 0.9999

    for _ in range(num_swaps):
        r1, c1 = np.random.randint(0, 2), np.random.randint(0, width)
        r2, c2 = np.random.randint(0, 2), np.random.randint(0, width)

        if (r1, c1) == (r2, c2) or m[r1, c1] == -1 or m[r2, c2] == -1: continue

        e_before = get_local_cost(r1, c1, m, distances) + get_local_cost(r2, c2, m, distances)

        m[r1, c1], m[r2, c2] = m[r2, c2], m[r1, c1]

        e_after = get_local_cost(r1, c1, m, distances) + get_local_cost(r2, c2, m, distances)

        delta = e_after - e_before

        if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
            curr_c += delta
        else:
            m[r1, c1], m[r2, c2] = m[r2, c2], m[r1, c1]

        T *= cooling

    # print(f"Final cost: {curr_c}")

    # Convert indices back to monster objects
    final_m = np.empty((2, width), dtype=object)
    for i in range(2):
        for j in range(width):
            idx = m[i, j]
            final_m[i, j] = monsters[idx] if idx != -1 else None

    return final_m, curr_c


def show_mosaic(m):
    h, w = m.shape
    baseline = 256

    v_step = int(baseline * 2.5)
    image = np.ones((h * v_step, w * baseline, 3), dtype=np.uint8) * 255

    for i in range(h):
        for j in range(w):
            if m[i, j] is not None:
                image[i * v_step:(i + 1) * v_step, j * baseline:(j + 1) * baseline, :] =  m[i, j].picture

    cv2.imwrite("visuals/mosaic.png", image)