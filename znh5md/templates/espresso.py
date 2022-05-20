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
