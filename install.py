"""Install pinsel with its Rust kernels from the latest GitHub release.

Picks the pinsel_core wheel for this machine and installs it with the pinsel
wheel into the Python that runs this script. Needs only the standard library
and pip; pulls in numpy and Pillow, nothing else.

    python install.py                (from a clone)
    curl -fsSL https://raw.githubusercontent.com/The-Chaos-Queen/Pinsel/main/install.py | python -

Read it first if you like; it is short.
"""
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request

API = 'https://api.github.com/repos/The-Chaos-Queen/Pinsel/releases/latest'

# (sys.platform prefix, machine names) -> tag in the pinsel_core wheel's file name
PLATFORMS = [
    ('linux', ('x86_64', 'amd64'), 'manylinux2014_x86_64'),
    ('linux', ('aarch64', 'arm64'), 'manylinux2014_aarch64'),
    ('win', ('amd64', 'x86_64'), 'win_amd64'),
    ('darwin', ('arm64', 'aarch64'), 'macosx_11_0_arm64'),
    ('darwin', ('x86_64',), 'macosx_10_12_x86_64'),
]


def fetch_json(url):
    req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json'})
    try:
        r = urllib.request.urlopen(req, timeout=30)
    except urllib.error.URLError as e:
        if 'CERTIFICATE_VERIFY_FAILED' not in str(e):
            raise
        # some standalone Pythons ship without CA certificates: use pip's bundle
        import ssl
        from pip._vendor import certifi
        r = urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context(cafile=certifi.where()))
    with r:
        return json.load(r)


def main():
    if sys.version_info < (3, 10):
        sys.exit('pinsel needs Python 3.10 or newer')
    machine = platform.machine().lower()
    tag = next((t for p, ms, t in PLATFORMS if sys.platform.startswith(p) and machine in ms), None)
    if tag is None:
        sys.exit(f'no pinsel_core wheel for {sys.platform}/{machine}; '
                 'install the numba engine instead: pip install "pinsel[numba] @ git+https://github.com/The-Chaos-Queen/Pinsel"')
    release = fetch_json(API)
    urls = {a['name']: a['browser_download_url'] for a in release['assets']}
    core = [u for n, u in urls.items() if n.startswith('pinsel_core-') and tag in n]
    pure = [u for n, u in urls.items() if n.startswith('pinsel-') and n.endswith('-py3-none-any.whl')]
    if not core or not pure:
        sys.exit(f'release {release.get("tag_name")} has no wheels for {tag}')
    print(f'pinsel {release["tag_name"]} for {tag}', flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', pure[0], core[0]])
    subprocess.check_call([sys.executable, '-c',
                           'import pinsel, pinsel._backend as B; print("pinsel", pinsel.__version__, "kernels:", B.NAME)'])


if __name__ == '__main__':
    main()
