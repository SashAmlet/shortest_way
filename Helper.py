
import random
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from shapely.wkt import loads


class Random:
    def generate_random_points(n, x_range, y_range):
        """Generating a list of random points."""
        return [Point(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]

    def cluster_points_to_polygons(points, h):
        """Clustering points and creating polygons."""
        # Перетворення списку об'єктів Point на масив для KMeans
        points_array = np.array([[point.x, point.y] for point in points])
        
        # Кластеризація точок
        kmeans = KMeans(n_clusters=h)
        clusters = kmeans.fit_predict(points_array)

        # Створення полігонів
        polygons = []
        for i in range(h):
            # Вибірка точок поточного кластера
            cluster_points = points_array[clusters == i]
            if len(cluster_points) < 3:
                continue  # Необхідно щонайменше 3 точки для створення полігону
            hull = ConvexHull(cluster_points)
            # Створення полігону з опуклої оболонки точок кластера
            polygons.append(Polygon(cluster_points[hull.vertices]))

        return polygons
    
class Painter:
    def draw_polygons(polygons, x_range, y_range, lines=None):
        """Drawing polygons and lines on a graph."""
        fig, ax = plt.subplots()
        for polygon in polygons:
            # Перетворення об'єкта Polygon на список координат для matplotlib
            mpl_polygon = MplPolygon(list(polygon.exterior.coords), closed=True, fill=None, edgecolor='r')
            ax.add_patch(mpl_polygon)

        # Перевіряємо, чи є lines списком ліній чи одиночною лінією
        if lines is not None:
            if isinstance(lines, list):  # Якщо lines – це список ліній
                for line in lines:
                    x_coords, y_coords = line.xy
                    line2D = Line2D(x_coords, y_coords, color='b')
                    ax.add_line(line2D)
            else:  # Якщо lines – це одиночна лінія
                x_coords, y_coords = lines.xy
                line2D = Line2D(x_coords, y_coords, color='b')
                ax.add_line(line2D)

        # Встановлення меж осі X та Y
        plt.xlim(x_range[0] - 50, x_range[1] + 50)
        plt.ylim(y_range[0] - 50, y_range[1] + 50)
        plt.show()


class File:
    def save_polygons_to_file(polygons, file_name):
        """Saving a list of polygons to a file"""
        # Відкриваємо файл для запису
        with open(file_name, 'w') as file:
            # Перебираємо всі багатокутники у списку
            for polygon in polygons:
                # Отримуємо WKT подання кожного Polygon
                wkt_representation = polygon.wkt
                # Записуємо WKT у файл, додаючи переклад рядка після кожного
                file.write(wkt_representation + '\n')

    def read_polygons_from_file(file_name):
        """Reading a list of polygons from a file"""
        polygons = []
        with open(file_name, 'r') as file:
            for line in file:
                # Пропускаємо порожні рядки
                if line.strip():
                    # Перетворимо рядок WKT на об'єкт Polygon і додаємо до списку
                    polygon = loads(line)
                    polygons.append(polygon)
        return polygons