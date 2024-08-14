## Computer Vision Pipeline Tools
cvpl_tools is a Python package for utilities and classes related to the file I/O, dataset record keeping for image processing and computer vision.

Functionalities this repository provides for managing datasets using DatasetReference:
- DatasetReference keeps an ordered array of image file paths
- Images are assigned unique IDs used to match with their annotation files
- DatasetReference can be written to disk as json using strenc.py, recording down the image files used in a processing step and their file paths
- cvpl_tools/fs.py provides interface to turn image files to and write them from numpy arrays
- reader_from_cmd.py defines a utility function for creating DatasetReference by pattern matching image files to be included in the dataset
- cvpl_tools/im folder provides an extensible segmentation pipeline that supports Napari visualization and intermediate result caching/checkpointing functionalities

### Dependency
This repository uses [Poetry](https://python-poetry.org/docs/) to manage dependency.