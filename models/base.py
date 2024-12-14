from abc import ABC, abstractmethod
import torch
import numpy as np

class PolicyBase(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def take_np_action(self, state: np.array):
        """Take Action from given state"""
        pass

    @abstractmethod
    def get_dist(self, state: np.array):
        """Get distribution from given state"""
        pass