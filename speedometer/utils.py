import cv2
import numpy as np

r = lambda x1, y1, x2, y2: np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def draw_str(dst, (x, y), s, p='l'):

    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA

    if p == 'r':
        scale, _ = cv2.getTextSize(s, font, 1.0, 2)
        x -= scale[0]
    elif p == 'm':
        scale, _ = cv2.getTextSize(s, font, 1.0, 2)
        x -= scale[0] / 2
    cv2.putText(dst, s, (x + 1, y + 1), font, 1.0, (0, 0, 0),
                thickness=2, lineType=line)
    cv2.putText(dst, s, (x, y), font, 1.0, (255, 255, 255),
                lineType=line)


def create_capture(source=0, **params):

    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        raise IOError('Unable to open video source: %s' % source)
    return cap


def create_writer(video_name, (x, y), fps=25, codec='MPEG'):

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(video_name, fourcc, fps, (x, y))
    if not out.isOpened():
        print 'Warning, cannot save file!'
        out = None
    return out


def scale(method):
    def wrapper(self, y, dy, *args, **kwargs):
        new_y = y / self.scale
        new_dy = dy / 100 + 0.5
        return method(self, *args, y=new_y, dy=new_dy, **kwargs)
    return wrapper


