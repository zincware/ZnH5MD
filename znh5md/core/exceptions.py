class GroupNotFound(Exception):
    """Raise if a Group does not exist in the H5MD file

    H5Py defaults to a KeyError. This Error is more specific
    """
