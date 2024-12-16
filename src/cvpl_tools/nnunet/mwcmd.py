"""
monai_wrapper command line utilities (named for legacy reasons, since this file has been moved
from monai_wrapper to cvpl_tools)
"""


import subprocess
import os


def run(args):
    print(f"Running args: {args}\ncommand:{' '.join(args)}")
    for vname in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        v = os.environ.get(vname)
        print(f'os.environ["{vname}"]={v}')
    process = subprocess.run(args, capture_output=True, text=True)
    print('---Printing stdout---\n', process.stderr)
    print('---Printing stderr---\n', process.stderr)
