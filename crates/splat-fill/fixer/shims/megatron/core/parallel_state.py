def is_initialized():
    return False


def model_parallel_is_initialized():
    return False


def get_data_parallel_world_size(*a, **k):
    return 1


def get_data_parallel_rank(*a, **k):
    return 0


def get_context_parallel_group(*a, **k):
    return None


def get_context_parallel_world_size(*a, **k):
    return 1


def get_data_parallel_group(*a, **k):
    return None
