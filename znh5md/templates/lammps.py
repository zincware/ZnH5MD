from znh5md.core import H5MDProperty


class LammpsH5MD:
    box = H5MDProperty(attribute="database", group="particles/all/box/edges")
    force = H5MDProperty(attribute="database", group="particles/all/force")
    image = H5MDProperty(attribute="database", group="particles/all/image")
    position = H5MDProperty(attribute="database", group="particles/all/position")
    species = H5MDProperty(attribute="database", group="particles/all/species")
    velocity = H5MDProperty(attribute="database", group="particles/all/velocity")

    def __init__(self, database):
        self.database = database
