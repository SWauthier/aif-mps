import numpy as np

try:
    import wandb
except:  # noqa: E722
    pass


class Log:
    """
    A logger construct that stores intermediate values during training in a
    dictionary of lists.
    """

    def __init__(self):
        self._logs = {}

    @property
    def logs(self):
        return self._logs

    def add(self, key, value, data_type):
        """
        Add a new log to the logger object
        :param key: Name of the value
        :param value: Actual value to log
        :param data_type: Type of data, can be "scalar" or "image"
        """
        prev = self._logs.get(key, None)
        old_value = []
        if prev is not None:
            old_value = prev["value"]
            if prev["type"] != data_type:
                # Incompatible data types
                raise ValueError

        self._logs[key] = {
            "value": old_value + [value],
            "type": data_type,
        }

    def to_writer(self, writer, epoch):
        """
        Store logs of this objects in a specific tensorboard writer
        :param writer: the writer object
        :param epoch: The epoch
        """
        for key, value in self._logs.items():
            if value["type"] == "scalar":
                # Average the mean value for this scalar over the entire epoch
                writer.add_scalar(key, np.mean(np.array(value["value"])), epoch)
            elif value["type"] == "image":
                # only show the last image, otherwise tensorboard becomes too large
                writer.add_image(key, value["value"][-1], epoch)
            elif value["type"] == "text":
                writer.add_text(key, value["value"][-1], epoch)
            elif value["type"] == "histogram":
                writer.add_histogram(key, np.array(value["value"]), epoch)

    def to_wandb(self, prefix):
        wandb_dict = {}
        for key, value in self._logs.items():
            if value["type"] == "scalar":
                # Average the mean value for this scalar over the entire epoch
                wandb_dict[f"{prefix}/{key}"] = np.mean(np.array(value["value"]))
            elif value["type"] == "image":
                wandb_dict[f"{prefix}/{key}"] = wandb.Image(value["value"][-1])
            elif value["type"] == "text":
                wandb_dict[f"{prefix}/{key}"] = wandb.Table(
                    data=[value["value"]], columns=[key] * len(value["value"])
                )
            elif value["type"] == "figure":
                wandb_dict[f"{prefix}/{key}"] = value["value"][-1]
        return wandb_dict

    def __add__(self, other):
        """
        Add two log objects together
        :param other: other log object
        """
        for k, v in other.logs.items():
            self.add(k, v["value"], v["type"])
        return self

    def __iadd__(self, other):
        """
        Add two log objects together
        :param other: other log object
        """
        return self + other
