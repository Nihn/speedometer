import numpy as np
import cv2

from collections import deque, defaultdict

from speedometer.utils import draw_str, create_capture, create_writer, r


class Parameter:
    def __init__(self, val):
        self.val = val


class App:

    window_name = 'Speedometer'
    spaces = 20
    lk_params = {'winSize': (21, 21),
                 'maxLevel': 3,
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03)}

    callbacks = defaultdict(lambda: lambda val, mod: val + mod, {
        'winSize': lambda val, mod: (val[0]+mod, val[1]+mod),
        'qualityLevel': lambda val, mod: np.abs(val + mod * 0.1),
        'speed_multi': lambda val, mod: np.abs(val + mod * 0.1),
        'criteria': lambda val, mod: (val[0], val[1] + mod, val[2] + mod * 0.01)
    })

    interface_index = 3

    def __init__(self, video_src=0, pos_x=0, pos_y=0, quality=0.3, damping=40,
                 speed_multi=0.2, save=''):

        tracks_number = 100

        self.track_len = 10
        self.detect_interval = 5
        self.prev_gray = None
        self.tracks = deque(maxlen=tracks_number)
        self.cam = create_capture(video_src)
        self.frame_idx = 0
        self.sum = deque()
        self.speed = 0
        self.last_speeds = deque([0], maxlen=damping)
        self.speed_multi = speed_multi

        _, frame = self.cam.read()
        height, width, _ = frame.shape

        self.start_x = 2 * width/5 + pos_x
        self.stop_x = 3 * width/5 + pos_x
        self.start_y = 4*height/5 + pos_y
        self.stop_y = height + pos_y
        self.bottom = height
        self.interface_top = self.bottom / 12

        self.feature_params = {
            'maxCorners': 20,
            'qualityLevel': quality,
            'minDistance': 7,
            'blockSize': 7
        }

        self.app_params = {
            'speed_multi': speed_multi,
            'tracks_number': tracks_number,
            'distance_mod': 1
        }

        self.out = None
        if save:
            fps = self.cam.get(cv2.CAP_PROP_FPS)

            self.out = create_writer(save, frame.shape[1], frame.shape[0], fps)

        cv2.namedWindow(self.window_name)
        self.clicked = False
        self.interface_clicked = False
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def on_mouse(self, event, x, y, *_):

        if y > self.interface_top and x < self.spaces * 12:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.interface_clicked = True
            elif event == cv2.EVENT_LBUTTONUP and self.interface_clicked:

                index_x = x // self.spaces
                index_y = (self.bottom - y) // self.spaces-self.interface_index

                self.interface_clicked = False
                key = (self.lk_params.keys() + self.feature_params.keys())[
                    index_y]

                if index_y == -2:
                    temp = self.app_params
                    key = 'speed_multi'
                elif index_y == -1:
                    temp = self.app_params
                    key = 'distance_mod'
                elif key in self.lk_params:
                    temp = self.lk_params
                else:
                    temp = self.feature_params

                if index_x == 1:
                    temp[key] = self.callbacks[key](temp[key], 1)
                elif index_x == 2:
                    temp[key] = self.callbacks[key](temp[key], -1)

            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
            self.start_x = self.stop_x = x
            self.start_y = self.stop_y = y
            self.tracks = deque(maxlen=self.app_params['tracks_number'])
        elif event == cv2.EVENT_LBUTTONUP:
            self.clicked = False
        elif self.clicked and event == cv2.EVENT_MOUSEMOVE:
            self.stop_x = x
            self.stop_y = y

    def run(self, skip):

        self.frame_idx = skip
        while skip:
            _ = self.cam.read()
            skip -= 1

        while True:

            _, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if self.tracks:

                sum_r = self._optical_flow(frame_gray, self.prev_gray)
                self._measure_speed(sum_r)

            if not self.frame_idx % self.detect_interval:
                self._get_new_tracks(frame_gray)

            self.prev_gray = frame_gray
            self._show_and_save(vis)
            self.frame_idx += 1

            key = cv2.waitKey(1)
            if 0xFF & key == 27:
                self._clean_up()
                break

    def _optical_flow(self, current_frame, previous_frame):

        p0 = np.reshape([tr[-1] for tr in self.tracks], (-1, 1, 2))
        p1 = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, p0, None,
                                      **self.lk_params)[0]
        p0r = cv2.calcOpticalFlowPyrLK(current_frame, previous_frame, p1, None,
                                       **self.lk_params)[0]
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = deque()
        sum_r = 0

        for tr, (x, y), good_flag in zip(
                self.tracks, p1.reshape(-1, 2), good):

            if not good_flag:
                continue

            tr.append((x, y))
            new_tracks.append(tr)

            # prevents > 0 speed when not moving
            sign = 1 if tr[-1][-1] - tr[0][-1] > 0 else -1
            radius = r(tr[0][0], tr[0][-1],
                       tr[-1][0], tr[-1][-1])
            height_mod = \
                (self.bottom - (tr[-1][-1] + tr[0][-1]) / 2) / self.bottom
            sum_r += sign * radius * self.app_params['speed_multi'] * \
                (1 + height_mod * self.app_params['distance_mod'])

        self.tracks = new_tracks
        return sum_r

    def _measure_speed(self, sum_r):
        if self.tracks:
            self.last_speeds.append(sum_r / len(self.tracks))
            self.speed = sum(self.last_speeds)/len(self.last_speeds)

    def _get_new_tracks(self, frame_gray):

        mask = np.zeros_like(frame_gray)
        mask[self.start_y:self.stop_y, self.start_x:self.stop_x] = 1

        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 5, 0)

        features = cv2.goodFeaturesToTrack(frame_gray, mask=mask,
                                           **self.feature_params)
        if features is not None:
            for x, y in features.reshape(-1, 2):
                # if y > frame.shape[0] / 2:
                self.tracks.append(deque([(x, y)],
                                         maxlen=self.track_len))

    def _show_and_save(self, vis):

        draw_str(vis, (self.spaces, self.spaces),
                 'track count: %d' % len(self.tracks))
        draw_str(vis, (self.spaces, self.spaces * 2),
                 'speed: %.2f km/h' % np.abs(self.speed))
        draw_str(vis, (self.spaces, self.spaces * 3),
                 'speed without dump: %.2f km/h' % np.abs(self.last_speeds[-1]))

        draw_str(vis, (self.spaces, self.bottom - self.spaces),
                 '+ - speed multi = %s' % self.app_params['speed_multi'])
        draw_str(vis, (self.spaces, self.bottom - self.spaces * 2),
                 '+ - distance mod = %s' % self.app_params['distance_mod'])
        for i, (key, var) in enumerate(self.lk_params.items() +
                                       self.feature_params.items(),
                                       self.interface_index):
            draw_str(vis, (self.spaces, self.bottom - i * self.spaces),
                     '+ - %s = %s' % (key, var))

        for tr in self.tracks:
            cv2.circle(vis, (tr[0][0], tr[0][-1]), 2, (255, 0, 0))
            cv2.circle(vis, (tr[-1][0], tr[-1][-1]), 2, (0, 0, 255))
            cv2.polylines(vis, [np.int32(tr)], False, (0, 255, 0))
        cv2.rectangle(vis, (self.start_x, self.start_y),
                      (self.stop_x, self.stop_y), (255, 0, 0))
        cv2.imshow(self.window_name, vis)

        if self.out is not None:
            self.out.write(vis)

    def _clean_up(self):
        if self.out is not None:
            self.out.release()
        self.cam.release()
        cv2.destroyAllWindows()
