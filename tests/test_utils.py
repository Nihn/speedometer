from unittest import TestCase
from mock import patch, call

from speedometer import utils
import cv2


class DrawStrTestCase(TestCase):

    def setUp(self):
        self.vis = 'foo'
        self.x = 1
        self.y = 2
        self.text = 'text'
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.line = cv2.LINE_AA
        self.call1_args = (self.font, 1.0, (0, 0, 0))
        self.call2_args = (self.font, 1.0, (255, 255, 255))

    @patch('speedometer.utils.cv2.putText')
    @patch('speedometer.utils.cv2.getTextSize')
    def test_draw_str_left(self, get_text_mock, put_text_mock):

        utils.draw_str(self.vis, (self.x, self.y), self.text)

        get_text_mock.assert_has_no_calls()
        put_text_mock.assert_has_calls([call(self.vis, self.text,
                                             (self.x + 1, self.y + 1),
                                             *self.call1_args, thickness=2,
                                             lineType=self.line),
                                        call(self.vis, self.text,
                                             (self.x, self.y),
                                             *self.call2_args,
                                             lineType=self.line)])

    @patch('speedometer.utils.cv2.putText')
    @patch('speedometer.utils.cv2.getTextSize')
    def test_draw_str_right(self, get_text_mock, put_text_mock):

        get_text_mock.return_value = (1, 2), None

        utils.draw_str(self.vis, (self.x, self.y), self.text, 'r')

        get_text_mock.assert_called_once_with(self.text, self.font, 1.0, 2)
        put_text_mock.assert_has_calls([call(self.vis, self.text,
                                             (self.x, self.y + 1),
                                             *self.call1_args, thickness=2,
                                             lineType=self.line),
                                        call(self.vis, self.text,
                                             (self.x - 1, self.y),
                                             *self.call2_args,
                                             lineType=self.line)])

    @patch('speedometer.utils.cv2.putText')
    @patch('speedometer.utils.cv2.getTextSize')
    def test_draw_str_middle(self, get_text_mock, put_text_mock):

        get_text_mock.return_value = (4, 6), None

        utils.draw_str(self.vis, (self.x, self.y), self.text, 'm')

        get_text_mock.assert_called_once_with(self.text, self.font, 1.0, 2)
        put_text_mock.assert_has_calls([call(self.vis, self.text,
                                             (self.x - 1, self.y + 1),
                                             *self.call1_args, thickness=2,
                                             lineType=self.line),
                                        call(self.vis, self.text,
                                             (self.x - 2, self.y),
                                             *self.call2_args,
                                             lineType=self.line)])


class CreateCaptureTestCase(TestCase):

    @patch('speedometer.utils.cv2.VideoCapture')
    def test_camera_source(self, capture_mock):

        cap = utils.create_capture()

        self.assertEqual(cap, capture_mock.return_value)
        capture_mock.assert_called_once_with(0)

    @patch('speedometer.utils.cv2.VideoCapture')
    def test_file_source(self, capture_mock):

        cap = utils.create_capture('fooo')

        self.assertEqual(cap, capture_mock.return_value)
        capture_mock.assert_called_once_with('fooo')

    @patch('speedometer.utils.cv2.VideoCapture')
    def test_capture_none(self, capture_mock):

        capture_mock.return_value = None

        with self.assertRaises(IOError) as cxt:
            utils.create_capture()
            self.assertEqual(cxt, 'Unable to open video source: 0')
        capture_mock.assert_called_once()

    @patch('speedometer.utils.cv2.VideoCapture')
    def test_capture_none(self, capture_mock):

        capture_mock.return_value.isOpened.return_value = False

        with self.assertRaises(IOError) as cxt:
            utils.create_capture()
            self.assertEqual(cxt, 'Unable to open video source: 0')
        capture_mock.assert_called_once()

    @patch('speedometer.utils.cv2.VideoCapture')
    def test_capture_with_size(self, capture_mock):

        w = cv2.CAP_PROP_FRAME_WIDTH
        h = cv2.CAP_PROP_FRAME_HEIGHT

        cap = utils.create_capture(size='20x10')

        capture_mock.assert_called_once_with(0)
        capture_mock.return_value.set.assert_has_calls([call(w, 20), call(h, 10)])
        self.assertEqual(cap, capture_mock.return_value)


class CreateWriterTestCase(TestCase):

    @patch('speedometer.utils.cv2.VideoWriter_fourcc')
    @patch('speedometer.utils.cv2.VideoWriter')
    def test_create_writer(self, writer_mock, fourcc_mock):

        file = 'foo'
        points = 1, 2
        out = utils.create_writer(file, points)

        fourcc_mock.assert_called_once_with(*'MPEG')
        writer_mock.assert_called_once_with(file, fourcc_mock.return_value,
                                            25, points)
        self.assertEqual(out, writer_mock.return_value)

    @patch('speedometer.utils.cv2.VideoWriter_fourcc')
    @patch('speedometer.utils.cv2.VideoWriter')
    def test_create_writer_custom_values(self, writer_mock, fourcc_mock):

        file = 'foo'
        points = 1, 2
        fps = 10
        codec = 'XVID'
        out = utils.create_writer(file, points, fps, codec)

        fourcc_mock.assert_called_once_with(*codec)
        writer_mock.assert_called_once_with(file, fourcc_mock.return_value,
                                            10, points)
        self.assertEqual(out, writer_mock.return_value)

    @patch('speedometer.utils.cv2.VideoWriter_fourcc')
    @patch('speedometer.utils.cv2.VideoWriter')
    def test_create_writer_file_not_open(self, writer_mock, fourcc_mock):

        file = 'foo'
        points = 1, 2
        writer_mock.return_value.isOpened.return_value = False

        out = utils.create_writer(file, points)

        fourcc_mock.assert_called_once_with(*'MPEG')
        writer_mock.assert_called_once_with(file, fourcc_mock.return_value,
                                            25, points)
        self.assertIsNone(out)
