from znh5md.core import H5MDProperty
from znh5md.templates.base import H5MDTemplate


class LammpsH5MD(H5MDTemplate):
    box = H5MDProperty(group="particles/all/box/edges")
    force = H5MDProperty(group="particles/all/force")
    image = H5MDProperty(group="particles/all/image")
    position = H5MDProperty(group="particles/all/position")
    species = H5MDProperty(group="particles/all/species")
    velocity = H5MDProperty(group="particles/all/velocity")
