'''
Usage: install after modifying the configurations: pip3 install -e .
'''

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

def get_requirements(path: str):
    return [l.strip() for l in open(path)]
setup(
       name='FightingGameAI',
       version='0.1',
       packages=find_packages(),
        install_requires=get_requirements("requirements.txt"),
       entry_points={
           'console_scripts': [
           ],
       },
       license='MIT',
       description='Fighting Game AI',
       long_description=open('README.md').read(),
       long_description_content_type='text/markdown',
       author='Your Name',
       author_email='neih4207@gmail.com',
       url='https://github.com/NeiH4207/FightingGameAI',
   )
