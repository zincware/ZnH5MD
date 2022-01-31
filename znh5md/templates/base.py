import pathlib
import typing

from znh5md.core.exceptions import GroupNotFound
from znh5md.core.h5md import H5MDProperty


class H5MDTemplate:
    def __init__(self, database: typing.Union[pathlib.Path, str]):
        self.database = database

    @property
    def database(self):
        """Enforce that an attribute with the name database exists"""
        return self._database

    @database.setter
    def database(self, value):
        self._database = value

    def get_groups(self) -> typing.List[str]:
        """Get all available groups in the provided file"""
        groups = []
        for name, group in vars(type(self)).items():
            if isinstance(group, H5MDProperty):
                try:
                    _ = getattr(self, name)[0]
                    groups.append(name)
                except GroupNotFound:
                    pass
        return groups
