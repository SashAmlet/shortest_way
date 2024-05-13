from shapely.geometry import Point, Polygon, LineString
import random
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
import math
from shapely.wkt import loads

random.seed(42)

def generate_random_points(n, x_range, y_range):
    """Генерация списка случайных точек."""
    return [Point(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]

def cluster_points_to_polygons(points, h):
    """Кластеризация точек и создание полигонов."""
    # Преобразование списка объектов Point в массив для KMeans
    points_array = np.array([[point.x, point.y] for point in points])
    
    # Кластеризация точек
    kmeans = KMeans(n_clusters=h)
    clusters = kmeans.fit_predict(points_array)

    # Создание полигонов
    polygons = []
    for i in range(h):
        # Выборка точек текущего кластера
        cluster_points = points_array[clusters == i]
        if len(cluster_points) < 3:
            continue  # Необходимо минимум 3 точки для создания полигона
        hull = ConvexHull(cluster_points)
        # Создание полигона из выпуклой оболочки точек кластера
        polygons.append(Polygon(cluster_points[hull.vertices]))

    return polygons

def draw_polygons(polygons, line=None):
    """Отрисовка полигонов на графике."""
    fig, ax = plt.subplots()
    for polygon in polygons:
        # Преобразование объекта Polygon Shapely в список координат для matplotlib
        mpl_polygon = MplPolygon(list(polygon.exterior.coords), closed=True, fill=None, edgecolor='r')
        ax.add_patch(mpl_polygon)

    # Если заданы координаты начала и конца линии, рисуем красную прямую
    if line:
        x_coords, y_coords = line.xy
        line2D = Line2D(x_coords, y_coords, color='b')
        ax.add_line(line2D)

    plt.xlim(x_range[0] - 50, x_range[1] + 50)  # Установка пределов оси X
    plt.ylim(y_range[0] - 50, y_range[1] + 50)  # Установка пределов оси Y
    plt.show()


def find_intersecting_polygons(polygons, line):
    """Определить все полигоны, которые пересекает прямая AB."""
    return [polygon for polygon in polygons if line.crosses(polygon)]

def find_closest_polygon(polygons, point):
    """Определить самый близкий полигон к заданной точке."""
    return min(polygons, 
                key=lambda polygon: point.distance(polygon),
                default=None)


def vector(p0,p1):
    return (p1[0]-p0[0], p1[1]-p0[1]) # без приведения к единичному
def norm(v):
    return (v[1], -v[0])
def dot_product(p0, p1):
    return p0[0] * p1[0] + p0[1] * p1[1]

def find_edge_points(polygon, line):
    """
    Находит две краевые точки полигона относительно линии.
    
    :param polygon: Объект Polygon, краевые точки которого нужно найти.
    :param line: Объект LineString, относительно которого ищутся точки.
    :return: Список из двух объектов Point, представляющих краевые точки.
    """
    # Вычисляем вектор для нашей прямой
    line_start, line_end = line.coords
    line_vector = vector(line_start, line_end)

    # Получаем список координат без последней точки, так как она дублирует первую
    coords = list(polygon.exterior.coords)[:-1] # упорядочены против часовой

    # Итерируемся по парам соседних точек, создавая сегменты
    intersecting_segments = []
    for current_point, next_point in zip(coords, coords[1:] + [coords[0]]):
        segment = LineString([current_point, next_point])
        if segment.intersects(line):
            intersecting_segments.append(segment)



    if not intersecting_segments:
        return None
    
    # Если нормаль сегмента сонаправлена с вектором прямой => скалярное умножение >=0 => данный сегмент дальний

    first_segment = None
    # Цикл по всем пересекающимся сегментам
    for segment in intersecting_segments:
        # Вычисляем вектор сегмента
        segment_vector = vector(segment.coords[0], segment.coords[1])
        # Нормализуем вектор сегмента
        segment_norm_vector = norm(segment_vector)
        # Проверяем условие
        if dot_product(segment_norm_vector, line_vector) < 0:
            first_segment = segment
            break

    # Проверяем, был ли найден первый сегмент
    if first_segment is None:
        raise ValueError("Не найден подходящий сегмент")


    # Задаем точку начала обхода
    start0_point = first_segment.coords[0]
    start1_point = first_segment.coords[1]

    # Находим индекс начальной точки
    start0_index = list(coords).index(start0_point)
    start1_index = list(coords).index(start1_point)

    edge_points = []
    if (start0_index and start1_index) is not None:
        # Обход многоугольника по часовой стрелке от заданной точки
        print("Walking clockwise from the point", start0_point)
        for i in range(start0_index-1, start0_index - len(coords), -1):
            new_point = coords[i % len(coords)]
            print(new_point)
            s0_vector = vector(new_point, start0_point)
            s0_norm_vector = norm(s0_vector)
            # если угол между прямой и нормалью сегмента <= 90 градусов => скалярное умножение >=0 => start0_point - угловая слева
            if dot_product(s0_norm_vector, line_vector) >=0:
                edge_points.append(start0_point)
                break
            else:
                start0_point = new_point

        # Обход многоугольника против часовой стрелки от заданной точки
        print("Walking counterclockwise from the point", start1_point)
        for i in range(start1_index+1, start1_index + len(coords)):
            new_point = coords[i % len(coords)]
            print(new_point)
            s1_vector = vector(start1_point, new_point)
            s1_norm_vector = norm(s1_vector)
            # если угол между прямой и нормалью сегмента <= 90 градусов => скалярное умножение >=0 => start1_point - угловая справа
            if dot_product(s1_norm_vector, line_vector) >=0:
                edge_points.append(start1_point)
                break
            else:
                start1_point = new_point
    else:
        print("Point not found in polygon.")
    return edge_points


def save_polygons_to_txt(polygons, file_name):
    # Открываем файл для записи
    with open(file_name, 'w') as file:
        # Перебираем все многоугольники в списке
        for polygon in polygons:
            # Получаем WKT представление каждого Polygon
            wkt_representation = polygon.wkt
            # Записываем WKT в файл, добавляя перевод строки после каждого
            file.write(wkt_representation + '\n')

def read_polygons_from_file(file_name):
    polygons = []
    with open(file_name, 'r') as file:
        for line in file:
            # Пропускаем пустые строки
            if line.strip():
                # Преобразуем строку WKT в объект Polygon и добавляем в список
                polygon = loads(line)
                polygons.append(polygon)
    return polygons

def SplitLine(polygons, line, position, step=1):
    #draw_polygons(polygons, line)
    
    coords = list(line.coords)
    
    start_point = Point(coords[position])
    segment = LineString(coords[position:position+2])

    intersecting_polygons = find_intersecting_polygons(polygons, segment)
    closest_polygon = find_closest_polygon(intersecting_polygons, start_point)

    while closest_polygon is not None:
        #draw_polygons([closest_polygon], line)
        p1, p2 = find_edge_points(closest_polygon, segment)
        print(f"\n\nEdge Points:\n{p1}, {p2}")

        coords.insert(position+1, p1)
        
        intr_polygons = find_intersecting_polygons(polygons, LineString([start_point, p1]))
        
        closst_polygon = find_closest_polygon(intr_polygons, start_point)
        line = LineString(coords)
        #draw_polygons(polygons, line)
        if closst_polygon is not None:
            line = SplitLine(polygons, line, position)
        
        line = SplitLine(polygons, line, position+step)
        
        break

    if position + 1 != len(line.coords) - 1:
        line = SplitLine(polygons, line, position+step)

    
    return line









# Initialization
n = 100  # Количество точек
h = 10    # Количество полигонов
x_range = (0, 100)  # Диапазон по оси X
y_range = (0, 100)  # Диапазон по оси Y

A = Point(-10, -10)
B = Point(110, 110)
line = LineString([A, B])



# Main part

points = generate_random_points(n, x_range, y_range)

polygons = read_polygons_from_file('polygon.txt')#cluster_points_to_polygons(points, h)#



splitLine = SplitLine(polygons, line, 0)



#closest_polygon = min(polygons, key=lambda polygon: start_point.distance(polygon))

#draw_polygons([closest_polygon], line)

#p1, p2 = find_edge_points(closest_polygon, line)











draw_polygons(polygons, splitLine)
#save_polygons_to_txt(polygons, 'polygon.txt')