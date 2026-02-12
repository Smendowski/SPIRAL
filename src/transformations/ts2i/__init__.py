def named(name: str):
    def decorator(fn):
        fn.pretty_name = name
        return fn

    return decorator
