from tools.visualize_components import Visualize_components
from utils.distance import components_distance
from monster.monster import Monster
import numpy as np
import os



if __name__ == "__main__":
    files = os.listdir("img")
    monsters: list[Monster] = []

    for file in files:
        monsters.append(Monster(file))

    for m1 in monsters:
        others = []
        for m2 in monsters:
            if m1 == m2: continue
            others.append((m2, components_distance(m1.centers, m1.weights, m2.centers, m2.weights)))
        others = sorted(others, key=lambda x: x[1])
        print(f"{m1.name}:\n \
        - {others[0][0].name} ({others[0][1]:.2f})\n \
        - {others[-1][0].name} ({others[-1][1]:.2f})")