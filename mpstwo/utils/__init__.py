import pathlib


def save_config(config, path):
    with open(path, "w") as fp:
        for kp, vp in config.items():
            if not isinstance(vp, dict):
                fp.write(f"{kp:<23}: {str(vp):>6}\n")
            else:
                fp.write(f"{kp}:\n")
                for key, value in vp.items():
                    fp.write(f"\t{key:<19}: {str(value):>6}\n")


def get_model_path(config):
    if hasattr(config, "model_path"):
        log_dir = pathlib.Path(config.model_path)
        if log_dir.is_dir() and log_dir.stem != "model":
            model_path = max(
                (log_dir / "model").glob("*-[0-9]*.pt"),
                key=lambda x: int(x.stem.split("-")[-1]),
            )
        elif log_dir.is_dir() and log_dir.stem == "model":
            model_path = max(
                log_dir.glob("*-[0-9]*.pt"),
                key=lambda x: int(x.stem.split("-")[-1]),
            )
            log_dir = log_dir.parent
        else:
            model_path = log_dir
            log_dir = log_dir.parent.parent
    else:
        log_dir = pathlib.Path(config.log_dir)
        model_path = max(
            (log_dir / "model").glob("*-[0-9]*.pt"),
            key=lambda x: int(x.stem.split("-")[-1]),
        )
    return model_path
