from Morphism import Morphism

class DummyMorphism(Morphism):

    def __init__(self, name):
        self.name = name

    def apply(self, data):
        print("this is morphism '{}'".format(self.name))
        return data + 1 # do nothing to the data
