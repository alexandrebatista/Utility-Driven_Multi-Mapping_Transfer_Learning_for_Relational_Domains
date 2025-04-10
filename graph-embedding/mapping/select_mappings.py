import random

def select_mappings(solutions):
    if len(solutions.get_robust_mappings()) <= 10:
        return solutions.get_robust_mappings()
    else:
        return random.sample(solutions.get_robust_mappings(),10)

def random_sampling_selection(solutions):
    if len(solutions.get_robust_mappings()) <= 10:
        return solutions.get_robust_mappings()
    else:
        return random.sample(solutions.get_robust_mappings(),10)

