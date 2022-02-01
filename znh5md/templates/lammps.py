"""ZnH5MD: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/
"""

from znh5md.core.h5md import H5MDProperty
from znh5md.templates.base import H5MDTemplate


class LammpsH5MD(H5MDTemplate):
    """Template for reading Lammps H5MD dump files

    Created with
    "dump   myDump all h5md  1000 NPT.lammpstraj position image velocity force species"
    """

    box = H5MDProperty(group="particles/all/box/edges")
    force = H5MDProperty(group="particles/all/force")
    image = H5MDProperty(group="particles/all/image")
    position = H5MDProperty(group="particles/all/position")
    species = H5MDProperty(group="particles/all/species")
    velocity = H5MDProperty(group="particles/all/velocity")
    not_exist = H5MDProperty(group="particles/all/sdfsd")
