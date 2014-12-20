import cv2
import numpy as np


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)


def create_capture(source=0, **params):

    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] += chunks[0] + ':'
        del chunks[0]

    source = chunks[0]
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
        print 'Warning: unable to open video source: ', source
    return cap


def create_writer(video_name, x, y, fps=25):

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(video_name, fourcc, fps, (x, y))
    if not out.isOpened():
        print 'Warning, cannot save file!'
    return out


def r(x1, y1, x2, y2):

    result = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if np.isnan(result):
        return 0
    return result
