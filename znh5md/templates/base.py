import pathlib
import typing


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
