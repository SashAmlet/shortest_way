class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

class Segment:
    def __init__(self, p=None, s=None, n=None, polygon=0):
        if p is None:
            p = [0, 0]
        if s is None:
            s = [0, 0]
        if n is None:
            n = Point()

        self.p = [int(num) for num in p]
        self.s = [int(num) for num in s]
        self.n = n
        self.polygon = int(polygon)

class Polygon:
    def __init__(self, segments=None):
        if segments is None:
            segments = []
        self.segments = segments

    def add_segment(self, segment):
        if isinstance(segment, Segment):
            self.segments.append(segment)
        else:
            raise TypeError("segment must be an instance of the Segment class")