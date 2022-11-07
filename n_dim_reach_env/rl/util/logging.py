"""This file describes the utility functions for logging.

Author: Jakob Thumm
Date: 4.11.2022
"""

from typing import Optional, Dict
from omegaconf.listconfig import ListConfig  # For hydra support


class Logger:
    """Logger class for internal logging."""

    def __init__(
        self,
        logging_keys: Optional[Dict] = {"reward": ["cum", "max"]}
    ):
        """Initialize the logger.

        The length is always logged.

        Args:
            logging_keys (Optional[Dict], optional): Keys to log. Defaults to {"reward": "cum"}.
                For every key, you can specify the type of logging. The following types are supported:
                - "cum": Cumulative logging.
                - "avg": Average logging.
                - "max": Maximum logging.
        """
        self.logging_keys = logging_keys
        for key in self.logging_keys:
            if not isinstance(self.logging_keys[key], (list, ListConfig)):
                self.logging_keys[key] = [self.logging_keys[key]]
            for v in self.logging_keys[key]:
                if v not in ["cum", "avg", "max"]:
                    raise ValueError(f"Unknown logging type {v} for key {key}.")
        self.reset()

    def reset(self):
        """Reset the logging info dictionary.

        Args:
            logging_keys (Optional[list], optional): Logging keys. Defaults to None.
        Returns:
            Dict[str, float]: Logging info dictionary.
        """
        self.logging_info = {
            "length": 0
        }
        for key in self.logging_keys:
            for v in self.logging_keys[key]:
                self.logging_info[f"{key}_{v}"] = 0
        return self.logging_info

    def log(
        self,
        info: Optional[Dict] = None,
        reward: Optional[float] = None,
    ):
        """Log the current information.

        Args:
            info (Optional[Dict], optional): Info to log. Defaults to None.
            reward (Optional[float], optional): Reward to log. Defaults to None.
        """
        self.logging_info["length"] += 1
        if info is None:
            info = {}
        if reward is not None:
            info["reward"] = reward
        for key, value in info.items():
            if key in self.logging_keys:
                value = float(value)
                if "cum" in self.logging_keys[key]:
                    self.logging_info[f"{key}_cum"] += value
                if "avg" in self.logging_keys[key]:
                    self.logging_info[f"{key}_avg"] = (self.logging_info[f"{key}_avg"] * (self.logging_info["length"]-1)
                                                       + value) / self.logging_info["length"]
                if "max" in self.logging_keys[key]:
                    self.logging_info[f"{key}_max"] = max(self.logging_info[f"{key}_max"], value)

    def get_logging_info(self) -> Dict[str, float]:
        """Get the logging info.

        Returns:
            Dict[str, float]: Logging info.
        """
        return self.logging_info
