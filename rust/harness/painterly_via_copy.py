"""Run examples/painterly_demo.py so that it imports the package from the current
directory (verify.sh's fastmath-off copy) instead of the editable install.

examples/painterly_demo.py has no sys.path line, so `python examples/painterly_demo.py`
imports the clone's pinsel (numba kernels with fastmath on), not the copy that
verify.sh made. Numba fastmath-on is not bit-comparable with Rust. This shim puts
the cwd first on sys.path. Use an absolute path to it:

    bash rust/harness/verify.sh $PWD/rust/harness/painterly_via_copy.py

The example itself runs unchanged and at full size (about 50 s per backend).
"""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import pinsel  # noqa: E402

assert os.path.dirname(os.path.abspath(pinsel.__file__)) == os.path.join(os.getcwd(), 'pinsel'), pinsel.__file__
sys.argv = ['painterly_demo.py']
runpy.run_path(os.path.join(os.getcwd(), 'examples', 'painterly_demo.py'), run_name='__main__')
