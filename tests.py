#!/usr/bin/env python

import unittest
import sys

from StringIO import StringIO

sys.stdout = StringIO()

testmodules = [
    'speedometer.tests.test_neural_network',
    'speedometer.tests.test_utils',
    ]

suite = unittest.TestSuite()

for t in testmodules:
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

unittest.TextTestRunner().run(suite)
