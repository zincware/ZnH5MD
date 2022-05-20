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
