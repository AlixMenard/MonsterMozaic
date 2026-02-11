from tools.visualize_components import Visualize_components
from tools.visualize_distances import Visualize_distances
from tools.mosaic import mosaic, show_mosaic
from utils.distance import components_distance
from monster.monster import Monster
import numpy as np
import os



if __name__ == "__main__":

    # collection = {
    #     "juice-mango-loco": 1,
    #     "juice-pipeline-punch": 7,
    #     "juice-viking-berry": 5,
    #     "juice-rio-punch": 6,
    #     "monster": 1,
    #     "monster-zero-sugar": 2,
    #     "rehab-peach-tea": 1,
    #     "top-speed-lando-norris": 1,
    #     "top-speed-vr46-zero-sugar": 1,
    #     "ultra-black": 1,
    #     "ultra-golden-pineapple": 2,
    #     "ultra-strawberry-dreams": 3,
    #     "ultra-violet": 3,
    #     "ultra-white": 10,
    # }
    # m = mosaic(collection)
    # show_mosaic(m)