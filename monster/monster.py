from utils.printer import print_colored_text
from sklearn.cluster import KMeans
from pathlib import Path
import numpy as np
import cv2

class Monster:

    def __init__(self, file:str, basesize:int = 256):
        self.path = Path("img") / file
        self.name = file.split(".")[0]
        self.image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        self.get_pixels(basesize)

    def get_pixels(self, basesize:int):
        # Get image in different spaces, filter alpha 0 out
        bgr = self.image[:, :, :3]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        alpha = self.image[:, :, 3]
        bgr[alpha == 0] = [0, 0, 0]

        # Get picture without background
        self.picture = bgr.copy()
        self.picture[alpha == 0] = [255, 255, 255]
        self.picture = cv2.resize(self.picture, (basesize, int(basesize*2.5)))
        self.half_picture = cv2.resize(self.picture, None, fx=0.5, fy=0.5)

        # Detect edges
        edges = cv2.Canny(bgr, 50, 150)
        kernel = np.ones((6, 6), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Extract relevant pixels
        non_edge_mask = (edges_dilated == 0).reshape((-1,))
        valid_mask = (alpha.reshape((-1,)) > 0) & non_edge_mask
        pixels = lab.reshape((-1, 3))[valid_mask]

        # Cluster pixels
        clusterer = KMeans(n_clusters=3)
        cluster_labels = clusterer.fit_predict(pixels)
        centers = clusterer.cluster_centers_
        weights = np.bincount(cluster_labels, minlength=3) / len(pixels)

        ordered_labels = np.argsort(weights)[::-1]
        self.centers = centers[ordered_labels]
        self.weights = weights[ordered_labels]
        self.weights = self.weights / np.sqrt(np.sum(self.weights**2))

    def print(self):
        print(f"--- {self.name} ---")
        for center, weight in zip(self.centers, self.weights):
            print_colored_text(f"{weight:.2f}", center)