import numpy as np
import cv2

from collections import deque, defaultdict

from speedometer.utils import draw_str, create_capture, create_writer
from speedometer.neural_network import NeuralNetwork


class App:

    window_name = 'Speedometer'
    spaces = 20
    training_corners_mod = 2

    callbacks = defaultdict(lambda: lambda val, mod: val + mod, {
        'winSize': lambda val, mod: (val[0]+mod, val[1]+mod),
        'qualityLevel': lambda val, mod: np.abs(val + mod * 0.1),
        'speed_multi': lambda val, mod: np.abs(val + mod * 0.01),
        'criteria': lambda val, mod: (val[0], val[1] + mod, val[2] + mod * 0.01)
    })

    def __init__(self, video_src=0, pos_x=0, pos_y=0, quality=0.3, damping=20,
                 speed_multi=0.02, save='', multiprocessed=False, epochs=1,
                 training_accuracy=20, training_length=40,
                 save_net='', load_net=''):
        """

        :param video_src: video source
        :param pos_x: initial x translation of capture window
        :param pos_y: initial y translation of capture window
        :param quality: quality used by goodFeaturesToTrack
        :param damping: speed = speeds[-damping:]/damping
        :param speed_multi: speed value multiplicator (only without neural net)
        :param save: if specified output file will be saved under this param
        :param multiprocessed: if true neural net training will be done in
        separated thread
        :param epochs: number of epochs for neural network
        :param training_accuracy: number of frames which will be training set
        for neural network
        :return: App instance
        """

        # Tracks related attributes
        tracks_number = 100
        self.track_len = 10
        self.tracks_count = 0
        self.detect_interval = 5
        self.tracks = deque([deque(maxlen=tracks_number)])
        ###########################

        # Speed measure realted attributes
        self.speed = 0
        self.last_speeds = deque([0], maxlen=damping)
        self.speed_multi = speed_multi
        ###################################

        # Neural network related attributes
        self.network_tuple = (2, 3, 1)
        self.training = None
        if load_net:
            self.network = NeuralNetwork(epochs=epochs, load=load_net,
                                         save=save_net)
        else:
            self.network = None
        self.multiprocessed = multiprocessed
        self.epochs = epochs
        self.save_net = save_net
        self.load_net = load_net
        self.samples = training_accuracy
        self.training_frames = training_length
        ####################################

        self.feature_params = {
            'maxCorners': 30,
            'qualityLevel': quality,
            'minDistance': 7,
            'blockSize': 7
        }
        self.lk_params = {
            'winSize': (21, 21),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         10, 0.03)}
        self.app_params = {
            'speed_multi': speed_multi,
            'tracks_number': tracks_number,
            'distance_mod': 1
        }

        # Frames related attributes
        self.prev_gray = None
        self.cam = create_capture(video_src)
        _, frame = self.cam.read()
        fps = self.cam.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1 / fps
        ############################

        # Interaface position related attributes
        height, width, _ = frame.shape
        params_len = (len(self.feature_params) +
                      len(self.lk_params) +
                      len(self.app_params))
        self.start_x = 2 * width/5 + pos_x
        self.stop_x = 3 * width/5 + pos_x
        self.start_y = 4*height/5 + pos_y
        self.stop_y = height + pos_y
        self.bottom = height
        self.right = width
        self.middle_x = width / 2
        self.interface = {
            'left': {
                'right_border': self.spaces * 3,
                'top_border': self.bottom - self.spaces * params_len,
                'clicked': False,
                # Depends on not feature or lk params in interface
                'index_mod': 3
            },
            'right': {
                'left_border': self.right - self.spaces * 4,
                'top_border': self.bottom - self.spaces * 2,
                'clicked': False
            }
        }
        self.clicked = False
        ############################################

        # Video capture related attributes
        self.out = None
        if save:
            self.out = create_writer(save, (frame.shape[1], frame.shape[0]),
                                     fps)
        ###################################

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, *_):

        if (x < self.interface['left']['right_border']
                and y > self.interface['left']['top_border']):
            self._interface('left', event, x, y)
        elif (x > self.interface['right']['left_border']
                and y > self.interface['right']['top_border']):
            self._interface('right', event)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Begin capturing new feature detection zone
            self.clicked = True
            self.start_x = self.stop_x = x
            self.start_y = self.stop_y = y
            self.tracks = deque(
                [deque(maxlen=self.app_params['tracks_number'])])
            ##############################################
        elif event == cv2.EVENT_LBUTTONUP:
            # End capturing new feature detection zone
            self.clicked = False
            ###########################################
        elif self.clicked and event == cv2.EVENT_MOUSEMOVE:
            # Capturing new feature detection zone
            self.stop_x = x
            self.stop_y = y
            ######################################

    def _interface(self, which, event, *args):

        if event == cv2.EVENT_LBUTTONDOWN:
                self.interface[which]['clicked'] = True
        elif (event == cv2.EVENT_LBUTTONUP
              and self.interface[which]['clicked']):
                method = getattr(self, '_%s_interface' % which)
                method(*args)

    def _left_interface(self, x, y):

        index_x = x // self.spaces
        index_y = ((self.bottom - y) // self.spaces -
                   self.interface['left']['index_mod'])

        self.interface['left']['clicked'] = False
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

    def _right_interface(self):

        self.network = NeuralNetwork(self.network_tuple,
                                     epochs=self.epochs,
                                     save=self.save_net)

        self.training = self.training_frames
        # use less features per zone durning training
        self.feature_params['maxCorners'] //= self.training_corners_mod
        self.app_params['tracks_number'] //= self.training_corners_mod

    def run(self, skip):

        frame_idx = skip
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

            if self.training:
                self.training -= 1
            elif not self.training and self.training is not None:

                if self.multiprocessed:
                    if self.network.is_done():
                        self.training = None
                        self._restore_limits_after_training()
                    elif not self.network.training.is_alive():
                        self.network.training.start()
                else:
                    self.training = None
                    self._restore_limits_after_training()
                    self.network.train()

            if not frame_idx % self.detect_interval:
                self._get_new_tracks(frame_gray)

            frame_idx += 1
            self.prev_gray = frame_gray
            self._show_and_save(vis)

            key = cv2.waitKey(1)
            if 0xFF & key == 27:
                # quit on esc key
                self._clean_up()
                break
            elif key == 32:
                # pause on space
                cv2.waitKey()

    def _restore_limits_after_training(self):
        self.feature_params['maxCorners'] *= self.training_corners_mod
        self.app_params['tracks_number'] *= self.training_corners_mod

    def _optical_flow(self, current_frame, previous_frame):

        sum = 0
        expected_res = 0
        self.tracks_count = 0
        for i, track in enumerate(self.tracks):
            if track:
                p0 = np.reshape([tr[-1] for tr in track], (-1, 1, 2))
                p1 = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame,
                                              p0, None, **self.lk_params)[0]
                p0r = cv2.calcOpticalFlowPyrLK(current_frame, previous_frame,
                                               p1, None, **self.lk_params)[0]
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = deque(maxlen=self.app_params['tracks_number'])
                sum_r = 0

                for tr, (x, y), good_flag in zip(track, p1.reshape(-1, 2),
                                                 good):
                    if not good_flag:
                        continue

                    tr.append((x, y))
                    new_tracks.append(tr)

                    # pixel shift on y axis
                    radius = tr[-1][-1] - tr[-2][-1]
                    if self.training and not i:
                        self.network.add_sample(radius, tr[-1][0] - tr[-2][0],
                                                radius *
                                                self.app_params['speed_multi'])
                    elif self.training:
                        self.network.add_sample(radius, tr[-1][0] - tr[-2][0],
                                                expected_res)

                    if self.training is None and self.network:
                        # If neural network is ready use it
                        sum_r += self.network.result(radius,
                                                     tr[-1][0] - tr[-2][0])
                    else:
                        height_mod = \
                            10 * (self.bottom -
                                  (tr[-1][-1] + tr[0][-1]) / 2) / self.bottom
                        height_mod = (1 + height_mod) ** 2
                        sum_r += (radius * self.app_params['speed_multi'] *
                                  height_mod * self.app_params['distance_mod'])

                if self.training and not i:
                    # Result used to teaching neural network
                    expected_res = sum_r / len(track)

                self.tracks[i] = new_tracks
                self.tracks_count += len(self.tracks[i])
                sum += sum_r / len(track)
        return sum

    def _measure_speed(self, sum_r):
        if self.tracks:
            # measure speed based on frame duration and pixels movement
            self.last_speeds.append(sum_r / (self.frame_duration *
                                             len(self.tracks)))
            # speed is calculated as arythmetic average of last speeds
            self.speed = sum(self.last_speeds)/len(self.last_speeds)

    def _get_new_tracks(self, frame_gray):

        if self.training:
            # If there is active training features capture zone is divided into
            # small pieces where one on bottom is treated as teacher for the
            # rest
            self.tracks = deque(
                [deque([], maxlen=self.app_params['tracks_number'])]
                * self.samples)
            masks = deque()
            counter = self.samples

            while counter:
                # Get sub capture zones
                step = (self.stop_y - self.start_y) // self.samples
                stop_y = self.start_y + counter * step
                start_y = self.start_y + (counter - 1) * step
                self.t_points = (start_y, stop_y)
                masks.append(np.zeros_like(frame_gray))
                masks[-1][start_y:stop_y, self.start_x:self.stop_x] = 1
                counter -= 1
        else:
            self.tracks = deque(
                [deque([], maxlen=self.app_params['tracks_number'])])
            masks = deque([np.zeros_like(frame_gray)])
            masks[0][self.start_y:self.stop_y, self.start_x:self.stop_x] = 1

        for i, mask in enumerate(masks):
            # Dont get same tracks
            for x, y in [np.int32(tr[-1]) for tr in self.tracks[i]]:
                    cv2.circle(mask, (x, y), 5, 0)

            # Get new features
            features = cv2.goodFeaturesToTrack(frame_gray, mask=mask,
                                               **self.feature_params)
            if features is not None:
                # If new features detected add them to the rest
                for x, y in features.reshape(-1, 2):
                    self.tracks[i].append(deque([(x, y)],
                                                maxlen=self.track_len))

    def _show_and_save(self, vis):

        # Output - up left corner
        draw_str(vis, (self.spaces, self.spaces),
                 'track count: %d' % self.tracks_count)
        draw_str(vis, (self.spaces, self.spaces * 2),
                 'speed: %.2f km/h' % np.abs(self.speed))
        draw_str(vis, (self.spaces, self.spaces * 3),
                 'speed without dump: %.2f km/h' % np.abs(self.last_speeds[-1]))

        # Neural network training button - right bottom corner
        draw_str(vis, (self.right - self.spaces, self.bottom - self.spaces),
                 'train', 'r')

        # Program params - left bottom corner
        draw_str(vis, (self.spaces, self.bottom - self.spaces),
                 '+ - speed multi = %s' % self.app_params['speed_multi'])
        draw_str(vis, (self.spaces, self.bottom - self.spaces * 2),
                 '+ - distance mod = %s' % self.app_params['distance_mod'])
        for i, (key, var) in enumerate(self.lk_params.items() +
                                       self.feature_params.items(),
                                       self.interface['left']['index_mod']):
            draw_str(vis, (self.spaces, self.bottom - i * self.spaces),
                     '+ - %s = %s' % (key, var))

        # tracks
        if self.training is None:
            for track in self.tracks:
                for tr in track:
                    cv2.circle(vis, (tr[0][0], tr[0][-1]), 2, (255, 0, 0))
                    cv2.circle(vis, (tr[-1][0], tr[-1][-1]), 2, (0, 0, 255))
                    cv2.polylines(vis, [np.int32(tr)], False, (0, 255, 0))

        # rectangle where we tracking
        cv2.rectangle(vis, (self.start_x, self.start_y),
                      (self.stop_x, self.stop_y), (255, 0, 0))

        # Training progress indicator
        if self.training:
            percentage = 10 * (self.training_frames -
                               self.training) // self.training_frames
            draw_str(vis, (self.middle_x - self.spaces, self.spaces),
                     'Getting samples...', 'm')
            draw_str(vis, (self.middle_x - self.spaces, self.spaces * 2),
                     '%s %d%%' % ('#' * percentage, percentage * 10), 'm')
        elif self.training is not None:
                draw_str(vis, (self.middle_x - self.spaces, self.spaces),
                         'Training...', 'm')
        cv2.imshow(self.window_name, vis)

        if self.out is not None:
            self.out.write(vis)

    def _clean_up(self):
        if self.out is not None:
            self.out.release()
        self.cam.release()
        cv2.destroyAllWindows()
