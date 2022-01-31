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


class EspressoH5MD(H5MDTemplate):
    """Template for reading Espresso H5MD dump files"""

    box = H5MDProperty(group="particles/atoms/box/edges")
    charge = H5MDProperty(group="particles/atoms/charge")
    force = H5MDProperty(group="particles/atoms/force")
    id = H5MDProperty(group="particles/atoms/id")
    image = H5MDProperty(group="particles/atoms/image")
    mass = H5MDProperty(group="particles/atoms/mass")
    position = H5MDProperty(group="particles/atoms/position")
    species = H5MDProperty(group="particles/atoms/species")
    velocity = H5MDProperty(group="particles/atoms/velocity")
