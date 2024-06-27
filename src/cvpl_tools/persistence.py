"""
This file handles file persistence of common objects used in pipeline. For example, dataset references
that specifies the files in a dataset can be stored as a file and read using the interfaces in this
file. Note that many file format is a directory containing a single file, which may seem that the directory
is unnecessary but this is to accommodate for the possible need to add more files to store information
about the object class' instance.
"""
import copy
import shutil
import json
from cvpl_tools.fs import ensure_dir_exists
from cvpl_tools.strenc import get_encoder, get_decoder_hook
from cvpl_tools.dataset_reference import DatasetReference


def write_dataset_reference(ref: DatasetReference, path: str):
    ensure_dir_exists(path, True)
    with open(f'{path}/dataset.json', 'w') as outfile:
        json.dump(ref, outfile, cls=get_encoder(), indent=2)


def read_dataset_reference(path: str) -> DatasetReference:
    with open(f'{path}/dataset.json', 'r') as infile:
        ref = json.load(infile, object_hook=get_decoder_hook())
    return ref


def write_dict(d, path: str, large_files=None):
    if large_files is None:
        large_files = {}
    ensure_dir_exists(path, True)
    d = copy.copy(d)
    for k in large_files:
        cur_path = d[k]
        target_path = large_files[k]
        if cur_path is None:
            raise ValueError(f'ERROR: Large file intended to be saved is not found from d[{k}]')
        d[k] = target_path
        shutil.copy(cur_path, target_path)
    with open(f'{path}/model_config.json', 'w') as outfile:
        json.dump(d, outfile, cls=get_encoder(), indent=2)


def read_dict(path: str):
    with open(f'{path}/model_config.json', 'r') as infile:
        d = json.load(infile, object_hook=get_decoder_hook())
    return d
