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
        self.picture = bgr.copy()
        bgr = cv2.blur(bgr, (5,5))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        alpha = self.image[:, :, 3]
        bgr[alpha == 0] = [0, 0, 0]

        # Get picture without background
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
        clusterer = KMeans(n_clusters=3, n_init=10)
        cluster_labels = clusterer.fit_predict(pixels)

        self.pixels, self.cluster_labels, self.centers = self.refine_clusters(pixels, cluster_labels,
                                                                              clusterer.cluster_centers_)

        weights = np.bincount(cluster_labels, minlength=3) / len(pixels)
        ordered_labels = np.argsort(weights)[::-1]

        self.centers = self.centers[ordered_labels]
        self.weights = weights[ordered_labels]
        self.weights = self.weights / self.weights.sum()

    def refine_clusters(self, pixels, labels, centers, sigma_mult=2.0):
        refined_pixels = []
        refined_labels = []

        for i in range(len(centers)):

            idx = np.where(labels == i)[0]
            cluster_points = pixels[idx]
            center = centers[i]

            distances = np.linalg.norm(cluster_points - center, axis=1)

            avg_dist = np.mean(distances)
            std_dist = np.std(distances)

            threshold_dist = avg_dist + std_dist * sigma_mult
            keep_indices = idx[distances <= threshold_dist]

            refined_pixels.append(pixels[keep_indices])
            refined_labels.append(labels[keep_indices])

        return np.vstack(refined_pixels), np.concatenate(refined_labels), centers

    def print(self):
        print(f"--- {self.name} ---")
        for center, weight in zip(self.centers, self.weights):
            print_colored_text(f"{weight:.2f}", center)