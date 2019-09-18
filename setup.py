from setuptools import find_packages, setup
from pathlib import Path

NAME = "rlsort"
VERSION = "0.1.0"

def set_init_dir():
    symlink_dict = {
        f'lib/{NAME}': 'src'
    }

    for sym_name, tgt_name in symlink_dict.items():
        sym_p = Path(sym_name)
        tgt_p = Path(tgt_name)
        if sym_p.exists() or not tgt_p.exists():
            continue
        for i in range(len(sym_p.parts) - 1):
            tgt_p = Path('../').joinpath(tgt_p)
        print('make symbolic link {} to {}'.format(sym_p, tgt_p))
        sym_p.symlink_to(tgt_p)

set_init_dir()
setup(name='rlsort',
      version='0.1.0',
      author='Hiro Inoue',
      license='MIT License',
      packages=find_packages('lib'),
      package_dir={'': 'lib'},
)
