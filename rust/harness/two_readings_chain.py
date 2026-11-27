"""Run examples/untitled_marbling.py, then examples/two_readings.py, in one process.

two_readings reads output/untitled_marbling.png, and verify.sh empties output/
before every script, so verify it through this file:

    bash rust/harness/verify.sh rust/harness/two_readings_chain.py
"""
import os
import runpy
import sys

ex = os.path.join(os.getcwd(), 'examples')   # verify.sh: cwd is its fastmath-off copy
sys.path.insert(0, os.getcwd())
sys.path.insert(0, ex)                       # two_readings imports untitled_marbling
runpy.run_path(os.path.join(ex, 'untitled_marbling.py'), run_name='__main__')
runpy.run_path(os.path.join(ex, 'two_readings.py'), run_name='__main__')
