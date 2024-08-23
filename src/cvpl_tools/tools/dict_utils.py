"""
This file defines some additional utilities function for Python dict class, especially for the
purpose of passing parameter settings hierarchically into sub-functions

For completeness, some functions are Python built-in and are not defined here:
dictionary merge (overwrite, to be precise): dictA | dictB
"""
from typing import Iterable, Any


def subset(d: dict, key_set: Iterable, inplace=False) -> dict:
    """Removes keys in key_set from d

    Args:
        d: The input dictionary
        key_set: Set of keys to remove
        inplace: If True, the operation is performed in place

    Returns:
        The result dictionary; if key_set is ordered, then the result dictionary's keys are ordered in
        that order as well.
    """
    if inplace:
        key_set = set(key_set)
        for key in d:
            if key not in key_set:
                d.pop(key)
        for key in key_set:
            assert key in d, f'Subset key {key} does not exist in dictionary d!'
        return d
    else:
        return {key: d[key] for key in key_set}


def add(d1: dict, d2: dict, inplace=False, allow_overlap=False):
    if not allow_overlap:
        overlap = set(d1.keys()) & set(d2.keys())
        assert overlap == set(), f'Key overlapped: {overlap}'
    if inplace:
        d1 |= d2
        return d1
    else:
        return d1 | d2


def minus(d: dict, key_set, inplace=False, allow_missing=True):
    """Removing a set of keys from dictionary

    Args:
        d: The original dictionary
        key_set: Keys to be removed
        inplace: If True, remove the keys from the original dict; otherwise return a new dict
        allow_missing: If False, will throw an error if key_set has a key not found in d

    Returns:
        The resulting dictionary
    """
    if inplace:
        for key in key_set:
            if key in d:
                d.pop(key)
            else:
                assert allow_missing, f'The key to be subtracted {key} is missing from the dictionary d!'
        return d
    else:
        key_set = set(key_set)
        nminus = len(key_set)
        result = {}
        for key in d:
            if key not in key_set:
                result[key] = d[key]
            else:
                nminus -= 1
        assert allow_missing or nminus == 0, f'Some keys to be subtracted are not found in the dictionary'
        return result


def minus_single(d: dict, key: Any, inplace=False, allow_missing=True):
    """Similar to minus(), but with only one key to be subtracted"""
    if inplace:
        if key in d:
            d.pop(key)
        else:
            assert allow_missing, f'Key {key} not found in dictionary!'
        return d
    else:
        result = {key: value for key, value in d.items()}
        if key in result:
            result.pop(key)
        else:
            assert allow_missing, f'Key {key} not found in dictionary!'
        return result
