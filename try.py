import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString

from Helper import File

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString
from MagicBall import MagicBall

def plot_polygons_and_wait_for_clicks(polygons,x_range,y_range):
    fig, ax = plt.subplots()
    # Отображаем полигоны
    for poly in polygons:
        mpl_poly = MplPolygon(list(poly.exterior.coords), closed=True, edgecolor='r', fill=False)
        ax.add_patch(mpl_poly)

    points = []  # Список для хранения точек нажатия
    lines = []   # Список для хранения объектов линий

    def onclick(event):
        # Добавляем точку (x, y) в список
        points.append((event.xdata, event.ydata))
        # Если точек четное количество, рисуем линию
        if len(points) % 2 == 0:
            line = LineString(points[-2:])  # Создаем линию из последних двух точек
            x, y = line.xy
            line_obj = ax.plot(x, y, color='blue')  # Отрисовываем линию на графике
            mb = MagicBall(line, polygons)
            shortest_line = mb.get_shortest_line()
            x1, y1 = shortest_line.xy
            lines_obj = ax.plot(x1, y1, color='black')
            lines.append(line_obj)
            lines.append(lines_obj)
            plt.draw()
        # Если это третий клик, удаляем последнюю линию
        if len(points) % 3 == 0:
            last_line = lines.pop()  # Удаляем последнюю линию из списка
            last_line.remove()       # Удаляем линию с графика
            plt.draw()

        # def toggle_onclick(event):
        #     global button_state
        #     button_state = not button_state
        #     if button_state:
        #         print("The onclick method is activated.")
        #     else:
        #         print("The onclick method is disabled.")

        # # Устанавливаем положение кнопки
        # button_ax = plt.axes([0.8, 0.0, 0.1, 0.05])
        # button = Button(button_ax, 'Вкл/Выкл')

        # # Привязываем функцию к событию
        # button.on_clicked(toggle_onclick)
        # # Привязываем onclick к событию клика на графике, если кнопка активирована
        # fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event) if button_state else None)
        # Зв'язуємо подію кліку мишею з функцією onclick
    # Связываем событие клика мышью с функцией onclick
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.xlim(x_range[0] - 50, x_range[1] + 50)
    plt.ylim(y_range[0] - 50, y_range[1] + 50)
    plt.show()

# Пример использования функции
# polygons - это список объектов shapely.geometry.Polygon

# Пример использования функции
# polygons - это список объектов shapely.geometry.Polygon
x_range = (0, int(10e2))  # Діапазон по осі X
y_range = (0, int(10e2))  # Діапазон по осі Y
polygons = File.read_polygons_from_file('polygon.txt')
plot_polygons_and_wait_for_clicks(polygons,x_range,y_range)
