"""
This file defines DatapointReference and DatasetReference

A dataset reference consists of many data point references. Its job is to make
clear what data files are in the dataset and how they should be read into the
memory (when the dataset is used). A dataset reference can therefore be used
to pass data into a workload, and the reference can be stored as a file afterward
to make the workload reproducible.
"""


from datetime import datetime
from cvpl_tools.array_key_dict import ArrayKeyDict
import cvpl_tools.fs as fs
import numpy as np
from dataclasses import dataclass


@dataclass
class DatapointReference:
    """
    Represents a pointer to location of data.
    The data is either an image or a chunk (data point) in the dataset, and one such
    datapoint can always be represented as a single numpy array.
    """
    data_ref: str | list

    def __init__(self, data_ref: str | list):
        """
        Create a DatapointReference object
        Args:
            data_ref: A list of strings OR a single string containing link to one or more files referred to by
                this datapoint reference
        """
        assert isinstance(data_ref, (str, list)), 'ERROR: Datapoint reference should be either a string or a list!'
        self.data_ref = data_ref

    def __str__(self):
        """
        Returns:
            A readable string representation of the link(s)
        """
        return str(self.data_ref)

    def ref(self):
        """
        Returns:
            A list of strings OR a single string containing link to one or more files referred to by
            this datapoint reference
        """
        return self.data_ref

    def has_multiple_images(self):
        """
        Returns:
            True if the datapoint is described by multiple images (in which case there is a list of
            links pointing to these images)
        """
        return isinstance(self.data_ref, list)

    def read_as_np(self, read_setting: fs.ImReadSetting) -> np.ndarray:
        """
        Reads the datapoint as a numpy array
        Args:
            read_setting: An cvpl_tools.fs.ImReadSetting object that describes how an image file is
            readed into memory into a numpy array
        Returns:
            The readed numpy array
        """
        return fs.ImIO.read_single_image(read_setting, self.data_ref)[0]


@dataclass
class DatasetReference:
    """
    This class' instance represents a dataset, and whose attributes include information on what data is in the dataset
    and when and how this reference is created.

    The datapoints in a dataset should be homogeneous - meaning they can be processed in the same way, and images
    should be stored in a separate dataset as its corresponding labels
    """
    datapoint_refs: ArrayKeyDict[str, DatapointReference]
    im_read_setting: fs.ImReadSetting
    creation_date: datetime
    creation_info: str  # a description of how this dataset is created
    name: str  # name of the dataset reference

    @staticmethod
    def new(datapoint_refs: ArrayKeyDict[str, DatapointReference],
            dataset_name: str,
            creation_info: str = '',
            im_read_setting: fs.ImReadSetting = None):
        """
        The recommended interface to create a DatasetReference object.
        Args:
            datapoint_refs: Mapping datapoint ids (imids) to DatapointReference objects
            dataset_name: Name of the dataset
            creation_info: A description of how the dataset is created; for better reproducibility archiving
            im_read_setting: A setting object that will be used to read images into memory (if needed)

        Returns:

        """
        return DatasetReference(
            datapoint_refs=datapoint_refs,
            creation_date=datetime.now(),
            creation_info=creation_info,
            name=dataset_name,
            im_read_setting=im_read_setting
        )

    @staticmethod
    def empty():
        """
        Create an empty DatasetReference object
        Returns:
            The created DatasetReference object
        """
        return DatasetReference.new(ArrayKeyDict(),
                                'Empty Dataset',
                                'This dataset is created as an empty dataset.',
                                None)


def test():
    in1 = 'C:/path/to/file1'
    r1 = DatapointReference(in1)
    in2 = ['C:/path/to/file2', 'C:/path/to/file3']
    r2 = DatapointReference(in2)
    assert str(r1) == 'C:/path/to/file1'
    assert len(r2.ref()) == 2 and r2.ref()[0] == 'C:/path/to/file2'
