import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon

points = []  # Список для зберігання точок натискання

def onclick(event):
    # Додаємо точку (x, y) у список
    points.append((event.xdata, event.ydata))
    # Якщо точок достатньо для створення полігону, рисуємо його
    if len(points) >= 3:
        poly = Polygon(points)  # Створюємо полігон зі списку точок
        mpl_poly = MplPolygon(list(poly.exterior.coords), closed=True, edgecolor='r', fill=False)
        ax.add_patch(mpl_poly)  # Відображаємо полігон на графіку
        plt.draw()
    # Якщо це четвертий клік, видаляємо останній полігон
    if len(points) % 4 == 0:
        points.pop()  # Видаляємо останню точку зі списку
        last_poly = ax.patches.pop()  # Видаляємо останній полігон зі списку
        last_poly.remove()  # Видаляємо полігон з графіку
        plt.draw()

# Створення графіку та прив'язка події onclick
fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
