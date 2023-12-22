def walk_modules(module, name="", depth=-1):
    
    child_list = list(module.named_children())

    if depth == 0 or len(child_list) == 0:
        yield (name, module)
    else:
        for child in child_list:
            yield from walk_modules(child[1], child[0] if name=="" else name + "." + child[0], depth - 1)