from Helper import Random, File
from MagicBall import MagicBall
from Helper import Painter

import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString

def plot_polygons_and_wait_for_clicks(polygons, x_range, y_range):
    fig, ax = plt.subplots()
    # Відображаємо полігони
    for poly in polygons:
        mpl_poly = MplPolygon(list(poly.exterior.coords), closed=True, edgecolor='r', fill=False)
        ax.add_patch(mpl_poly)

    points = []  # Список для зберігання точок натискання
    lines = []   # Список для зберігання об'єктів ліній
   

    def onclick(event):
        # Додаємо точку (x, y) до списку
        points.append((event.xdata, event.ydata))
        # Якщо точок парна кількість, малюємо лінію
        if len(points) % 2 == 0:
            line = LineString(points[-2:])  # Створюємо лінію з останніх двох точок
            x, y = line.xy
            line_obj = ax.plot(x, y, color='blue')  # Малюємо лінію на графіку

            mb = MagicBall(line, polygons)

            #####   PART 3 - FIND ALL PATHES   #####
            start_time = time.time()
            all_lines = mb.get_all_lines()
            #Painter.draw_polygons(polygons, x_range, y_range, all_lines)
            end_time = time.time()
            print(f"Paths are found in: {end_time - start_time} seconds")

            #####   PART 4 - FIND THE SHORTTEST PATH   #####
            start_time = time.time()
            shortest_line = mb.get_shortest_line()
            end_time = time.time()
            print(f"The shortest path is found found in: {end_time - start_time} seconds")

            x1, y1 = shortest_line.xy
            lines_obj = ax.plot(x1, y1, color='black')

            lines.append(line_obj)
            lines.append(lines_obj)
            plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.xlim(x_range[0] - x_range[1]/3, x_range[1] + x_range[1]/3)
    plt.ylim(y_range[0] - y_range[1]/3, y_range[1] + y_range[1]/3)
    plt.show()







if __name__ == "__main__":

    # Initialization
    n = int(10e2)  # Кількість точок
    h = int(100)    # Кількість полігонів
    x_range = (0, int(10e2))  # Діапазон по осі X
    y_range = (0, int(10e2))  # Діапазон по осі Y


    # Main part

    points = Random.generate_random_points(n, x_range, y_range)


    #####   PART 1 - BUILD POLYGONS   #####
    start_time = time.time()
    polygons = Random.cluster_points_to_polygons(points, h)#File.read_polygons_from_file('polygon1.txt')#
    end_time = time.time()

    File.save_polygons_to_file(polygons, 'polygon6.txt')
    execution_time = end_time - start_time
    print(f"Polygons are built in: {execution_time} seconds")

    #####   PART 2 - START ITERATIONS   #####
   
    button_state = False  # Глобальная переменная для отслеживания состояния

    plot_polygons_and_wait_for_clicks(polygons, x_range, y_range)