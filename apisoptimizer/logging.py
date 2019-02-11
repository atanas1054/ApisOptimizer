#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# logging.py (0.3.2)
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

try:

    from colorlogging import ColorLogger
    logger = ColorLogger(stream_level='disable')

except:

    from warnings import warn

    class NoLogger:

        def __init__(self):
            return

        def log(self, level, message, call_loc=None):
            return

        @property
        def stream_level(self):
            return None

        @stream_level.setter
        def stream_level(self, val):
            warn('ColorLogging is not installed')

        @property
        def file_level(self):
            return None

        @file_level.setter
        def file_level(self, val):
            warn('ColorLogging is not installed')

        @property
        def log_dir(self):
            return None

        @log_dir.setter
        def log_dir(self, val):
            warn('ColorLogging is not installed')

    logger = NoLogger()
