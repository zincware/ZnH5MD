import dataclasses
import os
import pathlib
import typing

from znh5md.format import FormatHandler

PATHLIKE = typing.Union[str, pathlib.Path, os.PathLike]


@dataclasses.dataclass
class H5MDBase:
    filename: PATHLIKE
    format_handler: FormatHandler = FormatHandler

    def __post_init__(self):
        self.format_handler = self.format_handler(self.filename)
