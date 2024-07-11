#!/usr/bin/env python
import numpy as np

class Network:
    def __init__(self, tensors):
        """
        Initialize a Network object.
        
        :param tensors: List of tensors representing the PEPS.
        """
        self.tensors = tensors
        
    def __repr__(self):
        return f"Network(tensors={self.tensors})"


class MPS(Network):


class PEPS(Network):
