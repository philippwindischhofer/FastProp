import functools

class Analysis:

    def __init__(self, name):
        self.name = name
        self.morphisms = [] # need to apply them in the correct order!

    def add_morphisms(self, morphs):
        for morph in morphs:
            self.add_morphism(morph)
        
    def add_morphism(self, morph):
        self.morphisms.append(morph) # only need to keep track of the actual 'apply' functions!

    def run(self, data):
        """
        Run the actual analysis, i.e. apply all morphisms in the intended order.
        """
        apply_analysis = functools.reduce(lambda f, g: lambda x: g(f(x)), self.morphisms, lambda x: x)
        print("now running analysis '{}'".format(self.name))
        return apply_analysis(data)
    
