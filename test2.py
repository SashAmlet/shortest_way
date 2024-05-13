from shapely.geometry import LineString, Polygon, Point

def find_intersecting_polygons(polygons, point_a, point_b):
    """Находит полигоны, которые пересекает прямая AB."""
    line_ab = LineString([point_a, point_b])
    return [polygon for polygon in polygons if line_ab.intersects(polygon)]

def find_extreme_points(polygons, line_ab):
    """Находит самые крайние точки в каждом полигоне относительно прямой AB."""
    extreme_points = []
    for polygon in polygons:
        # Преобразование полигона в линию для упрощения поиска точек
        poly_line = LineString(polygon.exterior.coords)
        # Находим точки пересечения
        intersection_points = line_ab.intersection(poly_line)
        # Сортируем точки пересечения по X координате
        sorted_points = sorted(intersection_points, key=lambda point: point.x)
        # Добавляем самую левую и правую точки
        extreme_points.append((sorted_points[0], sorted_points[-1]))
    return extreme_points

def build_paths(extreme_points, point_a, point_b):
    """Строит пути от A до B через выделенные точки."""
    paths = []
    for left_point, right_point in extreme_points:
        # Путь через левую точку
        path_left = LineString([point_a, left_point, point_b])
        # Путь через правую точку
        path_right = LineString([point_a, right_point, point_b])
        paths.append((path_left, path_right))
    return paths

# Пример использования
polygons = [Polygon([(0,0), (2,0), (1,2)]), Polygon([(3,0), (5,0), (4,2)])]
point_a = Point(0, 3)
point_b = Point(5, 3)

intersecting_polygons = find_intersecting_polygons(polygons, point_a, point_b)
extreme_points = find_extreme_points(intersecting_polygons, LineString([point_a, point_b]))
paths = build_paths(extreme_points, point_a, point_b)

for path_pair in paths:
    print(f"Путь через левую точку: {path_pair[0]}")
    print(f"Путь через правую точку: {path_pair[1]}")
