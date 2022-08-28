def list_of_list(x):
    if not isinstance(x, list):
        return False
    if not isinstance(x[0], list):
        return False
    return True
