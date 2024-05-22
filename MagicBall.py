from shapely.geometry import Point, LineString
from Helper import Painter

class Helper:
    @staticmethod
    def vector(p0,p1):
        return (p1[0]-p0[0], p1[1]-p0[1]) # без приведення до одиничного
    @staticmethod
    def norm(v):
        return (v[1], -v[0])
    @staticmethod
    def dot_product(p0, p1):
        return p0[0] * p1[0] + p0[1] * p1[1]


class MagicBall:
    def __init__(self, global_line, polygons):
        self.reached = False
        self.splitLines = []
        self.line = global_line
        self.polygons = polygons
        self.maxes = MagicBall.find_min_max(self)

    def find_intersecting_polygons(self, line):
        """Determine all the polygons that the line intersects."""
        return [polygon for polygon in self.polygons if line.crosses(polygon) or line.within(polygon)]

    @staticmethod
    def find_closest_polygon(polygons, point):
        """Determine the closest polygon to a given point."""
        return min(polygons, 
                    key=lambda polygon: point.distance(polygon),
                    default=None)
    
    @staticmethod
    def find_edge_points(polygon, line):
        """
        Finds the two edge points of a polygon relative to a line.
        
         :param polygon: The Polygon object whose edge points are to be found.
         :param line: LineString object relative to which points are searched.
         :return: A list of two Point objects representing the edge points.
        """
        # Обчислюємо вектор для нашої прямої
        line_start, line_end = line.coords
        line_vector = Helper.vector(line_start, line_end)

        # Отримуємо список координат без останньої точки, тому що вона дублює першу
        coords = list(polygon.exterior.coords)[:-1] # упорядочены против часовой

        # Ітеруємося по парах сусідніх точок, створюючи сегменти
        intersecting_segments = []
        for current_point, next_point in zip(coords, coords[1:] + [coords[0]]):
            segment = LineString([current_point, next_point])
            if segment.intersects(line):
                intersecting_segments.append(segment)



        if not intersecting_segments:
            return None
        
        # Якщо нормаль сегмента сонаправлена з вектором прямої => скалярне множення >=0 => даний сегмент дальній
        first_segment = None
        for segment in intersecting_segments:
            # Обчислюємо вектор сегмента
            segment_vector = Helper.vector(segment.coords[0], segment.coords[1])
            # Нормалізуємо вектор сегмента
            segment_norm_vector = Helper.norm(segment_vector)
            # Перевіряємо умову
            if Helper.dot_product(segment_norm_vector, line_vector) < 0:
                first_segment = segment
                break

        # Перевіряємо, чи було знайдено перший сегмент
        if first_segment is None:
            raise ValueError("No matching segment found")


        # Задаємо точку початку обходу
        start0_point = first_segment.coords[0]
        start1_point = first_segment.coords[1]

        # Знаходимо індекс початкової точки
        start0_index = list(coords).index(start0_point)
        start1_index = list(coords).index(start1_point)

        edge_points = []
        if (start0_index and start1_index) is not None:
            # Обхід багатокутника за годинниковою стрілкою від заданої точки
            #print("Walking clockwise from the point", start0_point)
            for i in range(start0_index-1, start0_index - len(coords), -1):
                new_point = coords[i % len(coords)]
                #print(new_point)
                s0_vector = Helper.vector(new_point, start0_point)
                s0_norm_vector = Helper.norm(s0_vector)
                # якщо кут між прямою та нормаллю сегмента <= 90 градусів => скалярне множення >=0 => start0_point - кутова зліва
                if Helper.dot_product(s0_norm_vector, line_vector) >=0:
                    edge_points.append(start0_point)
                    break
                else:
                    start0_point = new_point

            # Обхід багатокутника проти годинникової стрілки від заданої точки
            #print("Walking counterclockwise from the point", start1_point)
            for i in range(start1_index+1, start1_index + len(coords)):
                new_point = coords[i % len(coords)]
                #print(new_point)
                s1_vector = Helper.vector(start1_point, new_point)
                s1_norm_vector = Helper.norm(s1_vector)
                # якщо кут між прямою та нормаллю сегмента <= 90 градусів => скалярне множення >=0 => start1_point - кутова справа
                if Helper.dot_product(s1_norm_vector, line_vector) >=0:
                    edge_points.append(start1_point)
                    break
                else:
                    start1_point = new_point
        else:
            print("Point not found in polygon.")
        return edge_points

    def find_min_max(self):
        intersecting_polygons = self.find_intersecting_polygons(self.line)
        k=1.5
        l_distances = []
        r_distances = []
        
        for _pol in intersecting_polygons:
            p = self.find_edge_points(_pol, self.line)
            l_distances.append(self.line.distance(Point(p[0])))
            r_distances.append(self.line.distance(Point(p[1])))

        if l_distances == [] or r_distances == []:
            return None
        else:
            return (k*max(l_distances), k*max(r_distances))

    def to_optimize(self, coords, position, count):
        if position > 0:
            prev_point = Point(coords[position-1])
            next_point = Point(coords[position+1])
            new_segment = LineString([prev_point, next_point])
            
            if self.find_intersecting_polygons(new_segment) == []:
                new_coords = coords.copy()
                new_coords.remove(new_coords[position])
                
                (new_coords, count) = self.to_optimize(new_coords, position-1, count=count+1)
                return (new_coords, count)

        return (coords, count)

    def SplitLine(self, line, position, visited_polygons=[]):
        
        coords = list(line.coords)

        vpolygons = visited_polygons.copy()
        
        start_point = Point(coords[position])
        segment = LineString(coords[position:position+2])
        
        intersecting_polygons = self.find_intersecting_polygons(segment)
        closest_polygon = MagicBall.find_closest_polygon(intersecting_polygons, start_point)

        if closest_polygon is not None:
            p = MagicBall.find_edge_points(closest_polygon, segment)
            
            if not any(_polygon == closest_polygon for _polygon, i in vpolygons):
                cond = (0, 2)
            else:
                side = [i for _polygon, i in vpolygons if _polygon == closest_polygon]
                cond = (side[0], side[0]+1)

            for i in range(*cond):
                if self.line.distance(Point(p[i])) > self.maxes[i]:
                    continue
                
                self.reached = False
                new_coords = coords.copy()

                new_coords.insert(position+1, p[i])

                line = LineString(new_coords)

                index = next((index for index, (p, _) in enumerate(vpolygons) if p == closest_polygon), None)

                #Painter.draw_polygons(self.polygons, (0, int(10e2)), (0, int(10e2)), line)
                # Якщо кортеж знайдено, оновлюємо його значення i
                if index is not None:
                    vpolygons[index] = (closest_polygon, i)
                else:
                    # Якщо кортеж не знайдено, додаємо новий кортеж
                    vpolygons.append((closest_polygon, i))

                line = self.SplitLine(line, position, vpolygons)
        elif position > 0:
            (new_coords, count) = self.to_optimize(coords, position, 0)
            line = LineString(new_coords)
            position = position-count
 
        if position + 1 != len(line.coords) - 1 and not self.reached:
            line = self.SplitLine( line, position+1, vpolygons)
        if not self.reached:
            self.splitLines.append(line)
            self.reached = True
        return line

    def get_all_lines(self):
        if self.maxes is None:
            self.splitLines.append(self.line)
        else:
            self.SplitLine(self.line, 0)

        return self.splitLines
    
    def get_shortest_line(self):
        return min(self.splitLines, key=lambda line: line.length)









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