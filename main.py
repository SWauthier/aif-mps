import sys
from importlib import import_module


def parse_args(args: list[str]) -> dict:
    key = None
    value = None
    d = {}
    for s in args:
        if s.startswith("--"):
            key = s[2:]
            value = None
        else:
            value = s
        if key is not None and value is not None:
            d[key] = value
    return d


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    env = args[0]
    env = import_module(f"mpstwo.{env}")
    config = env.config
    kwargs = parse_args(args[2:])
    for k, v in kwargs.items():
        try:
            exec(f"config.{k} = {v}")
        except NameError:
            exec(f"config.{k} = '{v}'")
        except SyntaxError:
            exec(f"config.{k} = '{v}'")
    try:
        setattr(env, "_env", env.make_env(config))
        # env._env = env.make_env(config)
    except AttributeError:
        pass
    func = args[1]
    func = getattr(env, func)
    func(config)


if __name__ == "__main__":
    main(sys.argv[1:])
