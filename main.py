#!/usr/bin/env python

"""
Speedometer
====================

Uses Lucas-Kanade algorithm to evaluate train speed from front cam video.

Usage
-----
main.py [options]

Keys
----
ESC - exit
"""

import argh
import os

from speedometer.app import App


def start(video='', skip=0, pos_x=0, pos_y=0, quality=0.3, speed_multi=0.2,
          save='', multiprocessed=True, epochs=1):

    print __doc__

    if not video:
        video = 0
    elif not os.path.isfile(video):
        raise IOError('Wrong file name')

    App(video, pos_x, pos_y, quality, save=save, speed_multi=speed_multi,
        multiprocessed=multiprocessed, epochs=epochs).run(skip)


def main():
    argh.dispatch_command(start)

if __name__ == '__main__':
    main()
