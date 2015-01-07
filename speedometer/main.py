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

from app import App


def start(video='', skip=1, stop=None, pos_x=0, pos_y=0, quality=0.3,
          speed_multi=4, save='', multiprocessed=True, epochs=1,
          training_accuracy=20, training_length=40, save_net='', load_net='',
          max_net_error=0, tests=False):

    print __doc__

    if not video:
        video = 0
    elif not os.path.isfile(video):
        raise IOError('Wrong file name')

    App(video, pos_x, pos_y, quality, save=save, speed_multi=speed_multi,
        multiprocessed=multiprocessed, epochs=epochs,
        training_accuracy=training_accuracy, training_length=training_length,
        save_net=save_net, load_net=load_net,
        max_net_err=max_net_error).run(skip, stop=stop, tests=tests)


def main():
    argh.dispatch_command(start)

if __name__ == '__main__':
    main()
