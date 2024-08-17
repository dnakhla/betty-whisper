from setuptools import setup

APP = ['main.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'icon.icns',  # Replace with the path to your icon file if you have one
    'plist': {
        'CFBundleName': 'Betty',  # Set the app name here
        'LSUIElement': True,
    },
    'packages': ['certifi', 'rumps'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)