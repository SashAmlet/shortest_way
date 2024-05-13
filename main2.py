import numpy as np
import random
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from Polygon import Point, Segment, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D




def generate_random_points(n, x_range, y_range):
    return [Point(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]


def cluster_points_to_polygons(points, h):
    # Преобразование списка объектов Point в массив координат
    coords = np.array([[point.x, point.y] for point in points])
    # Кластеризация точек
    kmeans = KMeans(n_clusters=h)
    clusters = kmeans.fit_predict(np.array(coords))

    # Создание полигонов
    polygons = []
    for i in range(h):
        cluster_points = np.array([coords[j] for j in range(len(coords)) if clusters[j] == i])
        if len(cluster_points) < 3:
            continue  # Необходимо минимум 3 точки для создания полигона
        hull = ConvexHull(cluster_points)
        polygons.append((cluster_points[hull.vertices]))

    return polygons

def draw_polygons(polygons, line_start=None, line_end=None):
    fig, ax = plt.subplots()
    for polygon in polygons:
        # Преобразование объекта Polygon Shapely в список координат для matplotlib
        mpl_polygon = MplPolygon(polygon, closed=True, fill=None, edgecolor='b')
        ax.add_patch(mpl_polygon)

    # Если заданы координаты начала и конца линии, рисуем красную прямую
    if line_start and line_end:
        line = Line2D([line_start[0], line_end[0]], [line_start[1], line_end[1]], color='red')
        ax.add_line(line)

    plt.xlim(x_range[0] - 50, x_range[1] + 50)  # Установка пределов оси X
    plt.ylim(y_range[0] - 50, y_range[1] + 50)  # Установка пределов оси Y
    plt.show()






# Initization

n = 100  # Количество точек
h = 10    # Количество полигонов
x_range = (0, 100)  # Диапазон по оси X
y_range = (0, 100)  # Диапазон по оси Y
A = (-10, -10)
B = (110, 110)


# Main

points = generate_random_points(n, x_range, y_range)

polygons = cluster_points_to_polygons(points, h)
print(polygons)
draw_polygons(polygons, A, B)