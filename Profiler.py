class Profiler:

    def __init__(self, analysis):
        self.analysis = analysis

    def profile(self, data):
        """
        Profile the analysis, i.e. run it once and keep track of all
        the intermediate states. Saves all the information required
        for fast propagation.
        """

        # need to store the output of each morphism in the analysis, which forms the training data later on
        self.profile = {}
        cur_output = data
        for morph in self.analysis.morphisms:
            cur_name = morph.name
            cur_output = morph.apply(cur_output)
            self.profile[cur_name] = cur_output
