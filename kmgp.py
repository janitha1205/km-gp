import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

X, y = make_blobs(n_samples=500, n_features=2, centers=5, random_state=23)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:, 0], X[:, 1])
plt.show()


class kmeans:
    def __init__(self, k, data, itr):
        self.itr = itr
        self.k = k
        self.data = data
        self.clusters = {}
        center = np.random.rand(self.k + 1, self.data.shape[1])

        for i in range(self.k):
            points = []
            clus = {"center": center[i, :], "points": points}
            self.clusters[i] = clus

    def distance(self, p1, p2):
        n = len(p1)
        sum = 0
        for i in range(n):
            sum += (p1[i] - p2[i]) ** 2
        return np.sqrt(sum)

    def assign_clusters(self):
        for idx in range(self.data.shape[0]):
            dist = []

            curr_x = self.data[idx, :]

            for i in range(self.k):

                val = self.clusters[i]["center"]
                dis = self.distance(curr_x, val)
                dist.append(dis)
            curr_cluster = np.argmin(dist)
            self.clusters[curr_cluster]["points"].append([curr_x[0], curr_x[1]])

    def update_clusters(self):
        new_center = []
        for i in range(self.k):
            points = self.clusters[i]["points"]
            if len(points) > 0:

                mean_columns = [sum(col) / len(col) for col in zip(*points)]

                new_center.append(mean_columns)
        return new_center

    def algo(self):
        for i in range(self.itr):
            self.assign_clusters()
            new_c = self.update_clusters()
            print(new_c)
            sum_c = 0
            for i in range(self.k):
                sum_c += self.distance(self.clusters[i]["center"], new_c[i])
            if np.sqrt(sum_c) < 0.1:
                print(f"Converged after {i+1} iterations.")
                break
            for i in range(self.k):
                self.clusters[i]["center"] = new_c[i]

    def pred_cluster(self):
        pred = []
        for i in range(self.data.shape[0]):
            dist = []
            for j in range(self.k):
                val = self.clusters[j]["center"]

                dist.append(self.distance(self.data[i, :], self.clusters[j]["center"]))
            p = np.argmin(dist)
            pred.append(p)

        return pred


algo1 = kmeans(5, X, 100)
algo1.algo()
pre = algo1.pred_cluster()


def plotcurve(xc, yc, A):

    covariance = A

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        # Handle cases where covariance is a single value (isotropic)
        angle = 0
        width, height = 2 * np.sqrt(covariance), 2 * np.sqrt(covariance)

    a = 2 * width
    b = 2 * height

    phi = angle * np.pi / 180
    print(phi)
    t = np.linspace(0, 2 * np.pi, 100)

    x_unrot = a * np.cos(t)
    y_unrot = b * np.sin(t)

    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    Ell_unrot = np.array([x_unrot, y_unrot])
    Ell_rot = R @ Ell_unrot

    x_final = xc + Ell_rot[0, :]
    y_final = yc + Ell_rot[1, :]

    plt.plot(x_final, y_final, color="darkorange", linewidth=2)
    plt.scatter(xc, yc, color="red", marker="o", label="Center")  # Mark the center


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        # Handle cases where covariance is a single value (isotropic)
        angle = 0
        width, height = 2 * np.sqrt(covariance), 2 * np.sqrt(covariance)

    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=pre)
for i in range(algo1.k):
    center = algo1.clusters[i]["center"]
    po = algo1.clusters[i]["points"]

    plotcurve(center[0], center[1], np.cov(np.array(po).T))
    plt.scatter(center[0], center[1], marker="^", c="red")
plt.show()
