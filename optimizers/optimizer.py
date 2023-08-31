class Optimizer:

    def __init__(self, search_space: dict, obj_function = None):
        self.search_space = search_space
        self.hp_names = list(search_space.keys())
        self.obj_function = obj_function

    def suggest(self, n_iterations: int):
        raise NotImplementedError
