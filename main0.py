from shapely.geometry import Point, Polygon, LineString
from math import radians, cos, sin, atan2, pi
import numpy as np
import time

from Helper import Random, Painter, File


from rtree import index

import numba
from shapely.geometry import LineString, Polygon
import numpy as np
from MagicBall import MagicBall


#@numba.njit
# def find_intersecting_polygons(polygons, line):

#     # Угол поворота в радианах
#     line_start, line_end = line.coords
#     line_vector = vector(line_start, line_end)
#     angle_rad = atan2(line_vector[1], line_vector[0])

#     # вычисление угла относительно оси OY
#     angle_rad = pi/2 - angle_rad

#     # Матрица поворота
#     _cos = cos(angle_rad)
#     _sin = sin(angle_rad)
#     rotation_matrix = np.array([
#         [_cos, -_sin],
#         [_sin, _cos]
#     ])

#     rotated_line_с = apply_rotation(np.array(line.coords), rotation_matrix)
#     rotated_line = LineString(rotated_line_с)
#     Ax, Ay, Bx, By = rotated_line.bounds

#     intersecting_polygons = []
#     for _poly in polygons:
#         rotated_points = apply_rotation(np.array(_poly.exterior.coords), rotation_matrix)
#         min_x, min_y, max_x, max_y = np.min(rotated_points[:, 0]), np.min(rotated_points[:, 1]), np.max(rotated_points[:, 0]), np.max(rotated_points[:, 1])
#         if Ax > min_x and Ax < max_x and Ay < min_y and By > max_y:
#             intersecting_polygons.append(_poly)
    
    
#     return intersecting_polygons

# @numba.njit
# def apply_rotation(points, rotation_matrix):
#     return np.dot(points, rotation_matrix.T)

# def find_intersecting_polygons(polygons, line):
#     """Определить все полигоны, которые пересекает прямая AB."""

#     # Угол поворота в радианах
#     line_start, line_end = line.coords
#     line_vector = vector(line_start, line_end)
#     angle_rad = atan2(line_vector[1], line_vector[0])

#     # вычисление угла относительно оси OY
#     angle_rad = pi/2 - angle_rad

#     # Матрица поворота
#     _cos = cos(angle_rad)
#     _sin = sin(angle_rad)
#     rotation_matrix = np.array([
#         [_cos, -_sin],
#         [_sin, _cos]
#     ])

#     rotated_line_с = [rotation_matrix.dot(point) for point in line.coords]
#     rotated_line = LineString(rotated_line_с)

#     Ax, Ay, Bx, By = rotated_line.bounds


#     intersecting_polygons = []
#     for _poly in polygons:
#         # Применение поворота к каждой точке полигона
#         rotated_points = [rotation_matrix.dot(point) for point in _poly.exterior.coords]

#         # Создание нового полигона
#         rotated_polygon = Polygon(rotated_points)

#         # Получение новых значений bounds
#         min_x, min_y, max_x, max_y = rotated_polygon.bounds
#         if Ax > min_x and Ax < max_x and Ay < min_y and By > max_y:
#             intersecting_polygons.append(_poly)

#     return intersecting_polygons


def find_intersecting_polygons(polygons, line):
    """Определить все полигоны, которые пересекает прямая AB."""
    return [polygon for polygon in polygons if line.crosses(polygon) or line.within(polygon)]



def find_closest_polygon(polygons, point):
    """Определить самый близкий полигон к заданной точке."""
    return min(polygons, 
                key=lambda polygon: point.distance(polygon),
                default=None)

# from shapely.prepared import prep

# def find_intersecting_polygons(polygons, line):
#     """Определить все полигоны, которые пересекает прямая AB."""
#     prepared_line = prep(line)  # Подготавливаем линию для оптимизации операций
#     return [polygon for polygon in polygons if prepared_line.crosses(polygon) or prepared_line.within(polygon)]

# def find_closest_polygon(polygons, point):
#     """Определить самый близкий полигон к заданной точке."""
#     distances = {}  # Словарь для кэширования расстояний
#     for polygon in polygons:
#         # Вычисляем и кэшируем расстояние, если оно ещё не было вычислено
#         if polygon not in distances:
#             distances[polygon] = point.distance(polygon)
#     # Возвращаем полигон с минимальным расстоянием
#     return min(distances, key=distances.get, default=None)



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
        #print("Walking clockwise from the point", start0_point)
        for i in range(start0_index-1, start0_index - len(coords), -1):
            new_point = coords[i % len(coords)]
            #print(new_point)
            s0_vector = vector(new_point, start0_point)
            s0_norm_vector = norm(s0_vector)
            # если угол между прямой и нормалью сегмента <= 90 градусов => скалярное умножение >=0 => start0_point - угловая слева
            if dot_product(s0_norm_vector, line_vector) >=0:
                edge_points.append(start0_point)
                break
            else:
                start0_point = new_point

        # Обход многоугольника против часовой стрелки от заданной точки
        #print("Walking counterclockwise from the point", start1_point)
        for i in range(start1_index+1, start1_index + len(coords)):
            new_point = coords[i % len(coords)]
            #print(new_point)
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





def find_min_max(polygons, line):
    intersecting_polygons = find_intersecting_polygons(polygons, line)

    l_distances = []
    r_distances = []
    
    for _pol in intersecting_polygons:
        p = find_edge_points(_pol, line)
        l_distances.append(line.distance(Point(p[0])))
        r_distances.append(line.distance(Point(p[1])))


    return (2*max(l_distances), 2*max(r_distances))


def to_optimize(coords, position, polygons, count):
    if position > 0:
        prev_point = Point(coords[position-1])
        next_point = Point(coords[position+1])
        new_segment = LineString([prev_point, next_point])
        
        if find_intersecting_polygons(polygons, new_segment) == []:
            new_coords = coords.copy()
            new_coords.remove(new_coords[position])
            
            (new_coords, count) = to_optimize(new_coords, position-1, polygons, count=count+1)
            return (new_coords, count)

    return (coords, count)

def SplitLine(polygons, line, position, visited_polygons=[]):
    global reached, splitLines, global_line, maxes
    #draw_polygons(polygons, line)
    
    coords = list(line.coords)

    vpolygons = visited_polygons.copy()
    
    start_point = Point(coords[position])
    segment = LineString(coords[position:position+2])
    
    intersecting_polygons = find_intersecting_polygons(polygons, segment)
    closest_polygon = find_closest_polygon(intersecting_polygons, start_point)

    if closest_polygon is not None:
        #Painter.draw_polygons([closest_polygon], x_range, y_range, line)
        p = find_edge_points(closest_polygon, segment)
        #print(f"\n\nEdge Points:\n{p[0]}, {p[1]}")

        if not any(_polygon == closest_polygon for _polygon, i in vpolygons):
            cond = (0, 2)
        else:
            side = [i for _polygon, i in vpolygons if _polygon == closest_polygon]
            cond = (side[0], side[0]+1)

        for i in range(*cond):
            if global_line.distance(Point(p[i])) > maxes[i]:
                continue
            #if not any(_polygon == closest_polygon for _polygon, i in visited_polygons):
            reached = False
            new_coords = coords.copy()

            new_coords.insert(position+1, p[i])

            line = LineString(new_coords)
            # draw_polygons(polygons, [line])

            index = next((index for index, (p, _) in enumerate(vpolygons) if p == closest_polygon), None)


            # Если кортеж найден, обновляем его значение i
            if index is not None:
                vpolygons[index] = (closest_polygon, i)
            else:
                # Если кортеж не найден, добавляем новый кортеж
                vpolygons.append((closest_polygon, i))

            line = SplitLine(polygons, line, position, vpolygons)
    elif position > 0:
        (new_coords, count) = to_optimize(coords, position, polygons, 0)
        line = LineString(new_coords)
        position = position-count
        # if count > 0:
        #     draw_polygons(polygons, [line])



    #draw_polygons(polygons, [line])    
    if position + 1 != len(line.coords) - 1 and not reached:
        line = SplitLine(polygons, line, position+1, vpolygons)
    if not reached:
        splitLines.append(line)
        reached = True
    return line




if __name__ == "__main__":

    # Initialization
    n = int(10e4)  # Кількість точок
    h = 10    # Кількість полігонів
    x_range = (0, int(10e4))  # Діапазон по осі X
    y_range = (0, int(10e4))  # Діапазон по осі Y

    A = Point(-10, -10)
    B = Point(int(10e4+10), int(10e4+10))
    global_line = LineString([A, B])


    # Main part

    points = Random.generate_random_points(n, x_range, y_range)

    reached = False

    ############################
    start_time = time.time()
    polygons = Random.cluster_points_to_polygons(points, h)#File.read_polygons_from_file('polygon.txt')#
    end_time = time.time()

    #File.save_polygons_to_file(polygons, 'polygon2.txt')
    execution_time = end_time - start_time
    print(f"Polygons are built in: {execution_time} seconds")
    ###########################
    Painter.draw_polygons(polygons, x_range, y_range, global_line)

    maxes = find_min_max(polygons, global_line)

    splitLines = []
    ###########################
    start_time = time.time()
    splitLine = SplitLine(polygons, global_line, 0)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Paths are found in: {execution_time} seconds")
    ###########################
    start_time = time.time()
    shortest_line = min(splitLines, key=lambda line: line.length)
    # mb = MagicBall(global_line, polygons)
    # shortest_line = mb.get_shortest_line()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"The shortest path is found in: {execution_time} seconds")
    ###########################



    for line in splitLines:
        Painter.draw_polygons(polygons, x_range, y_range, line)

    #Painter.draw_polygons(polygons, x_range, y_range, splitLines)
    Painter.draw_polygons(polygons, x_range, y_range, shortest_line)