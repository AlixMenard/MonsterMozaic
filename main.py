from tools.visualize_components import Visualize_components
from tools.visualize_distances import Visualize_distances
from tools.mosaic import mosaic, show_mosaic, cost
from utils.distance import components_distance
from monster.monster import Monster
from time import time
import numpy as np
import os



if __name__ == "__main__":
    collection = {
        "java-cafe-latte": 0,           "java-french-vanilla": 0,          "java-irish-creme": 0,
        "java-loca-moca": 0,            "java-mean-bean": 0,               "java-mocha": 0,
        "java-salted-caramel": 0,       "juice-aussie-style-lemonade": 0,  "juice-bad-apple": 0,
        "juice-khaotic": 0,             "juice-mango-loco": 2,             "juice-mixxd": 0,
        "juice-monarch": 0,             "juice-pacific-punch": 0,          "juice-pipeline-punch": 7,
        "juice-rio-punch": 6,           "juice-ripper": 0,                 "juice-viking-berry": 5,
        "juice-voodoo-grape": 0,        "monster-electric-blue": 0,        "monster-lo-carb": 0,
        "monster-nitro-super-dry": 0,   "monster-orange-dreamsicle": 0,    "monster-strawberry-shot-zero-sugar": 0,
        "monster-strawberry-shot": 0,   "monster-super-premium-import": 0, "monster-zero-sugar": 3,
        "monster": 1,                   "rehab-green-tea": 0,              "rehab-peach-tea": 1,
        "rehab-strawberry-lemonade": 0, "rehab-tea-lemonade": 0,           "rehab-wild-berry-tea": 0,
        "reserve-orange-dreamsicle": 0, "reserve-peaches-creme": 0,        "top-speed-full-throttle": 0,
        "top-speed-lando-norris": 1,    "top-speed-vr46-zero-sugar": 1,    "top-speed-vr46": 0,
        "ultra-black": 1,               "ultra-blue-hawaiian": 0,          "ultra-blue": 0,
        "ultra-fantasy-ruby-red": 0,    "ultra-fiesta-mango": 0,           "ultra-golden-pineapple": 2,
        "ultra-paradise": 0,            "ultra-peachy-keen": 1,            "ultra-red": 0,
        "ultra-rosa": 0,                "ultra-strawberry-dreams": 3,      "ultra-sunrise": 0,
        "ultra-vice-guava": 0,          "ultra-violet": 3,                 "ultra-watermelon": 0,
        "ultra-white": 11,              "ultra-wild-passion": 0
    }

    monsters = np.array(list(collection.keys()))

    costs = []
    for i, monster in enumerate(monsters):
        print(f"{i+1}/{len(monsters)}")
        col = collection.copy()
        col[monster] += 1
        mos = []
        _, c = mosaic(col)
        costs.append(int(c))
    ind = np.argsort(costs)
    for monster in monsters[ind]:
        print(monster)