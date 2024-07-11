import numpy as np

class Node:
    """
    Class for nodes in MPS and PEPS tensor networks.

    Attributes:
    -----------
    name : string
        Name assigned to the node (should be unique)
    tensor : np.ndarray
        The tensor associated with this node.
    shape : tuple
        The shape of the node tensor
    spin_dim : float
        Spin dimension of lattice particles
    bond_dim : int
        Bond dimension of the node
    connected_nodes : dict
        Dictionary of connected nodes and indices connected

    Methods:
    --------
    __init__():
        Initialises a Node object.

    __repr__():
        Returns a string representation of the node.
    
    """

    def __init__(self, name, tensor, shape = None,
                 spin_dim = None, bond_dim = None,
                 connected_nodes = None):
        """
        Constructs a Node object.

        Parameters:
        -----------
        name : string
            Name assigned to the node (should be unique)
        tensor : np.ndarray
            The tensor associated with this node.
        shape : tuple, optional
            The shape of the node tensor
        spin_dim : float, optional
            Spin dimension of lattice particles
        bond_dim : int, optional
            Bond dimension of the node
        """
        self.name = name
        self.tensor = tensor
        if (shape and tensor.shape != shape):
            raise ValueError("Specified shape does not match tensor")

        self.connected_nodes = connected_nodes if connected_nodes else {}
        self.assign_shape(tensor.shape, spin_dim, bond_dim)

    def update_tensor(self, tensor):
        """
        Updates the tensor for a Node object, along with shape data

        Parameters:
        -----------
        tensor : np.ndarray
            The tensor associated with this node.
        """
        self.tensor = tensor
        self.assign_shape(tensor.shape, self.spin_dim, self.bond_dim)

    def assign_shape(self, shape, spin_dim = None, bond_dim = None):
        """
        Assigns shape to Node object (designed for private use)

        Parameters:
        -----------
        shape : tuple
            The shape of the node tensor
        spin_dim : float, optional
            Spin dimension of lattice particles
        bond_dim : int, optional
            Bond dimension of the node
        """
        self.shape = shape
        if (len(shape) < 2):
            self.spin_dim = shape[0]
            self.bond_dim = bond_dim
            return

        self.spin_dim = spin_dim if spin_dim else self.shape[0]
        self.bond_dim = bond_dim if bond_dim else self.shape[1]

    def __repr__(self):
        return (
               f"Node(name={self.name}, "
               f"shape={self.shape}, "
               f"spin dimension={self.spin_dim}, "
               f"bond dimension={self.bond_dim}, "
               f"connected nodes={self.connected_nodes})"
               )

    def contract(self, ax, other_node = None, new_name = None):
        """
        Contracts this node with another node.

        Parameters:
        -----------
        ax : tuple, array-like
            Axes of the tensor contraction
        other_node : Node, optional
            The other node to contract with.
        new_name : string, optional
            New name for created node, uses self.name by default
        
        Returns:
        --------
        Node
            A new node from the contraction operation
        """
        if (other_node and not isinstance(other_node, Node)):
            raise ValueError("other_node must be an instance of Node")

        if not new_name:
            new_name = self.name

        if (not other_node) or (other_node == self):
            return self.contract_self(ax, new_name)

        contracted_tensor = np.tensordot(self.tensor, other_node.tensor, axes = ax)
        return Node(new_name, contracted_tensor, connected_nodes = self.connected_nodes)

    def contract_self(self, ax, new_name = None):
        """
        Contracts this node with itself along an axis.

        Parameters:
        -----------
        ax : tuple
            Axes of the tensor contraction
        new_name : string, optional
            New name for created node, uses self.name by default
 
        Returns:
        --------
        Node
            A new node from the contraction operation
        """

        if len(ax) != 2:
            raise ValueError("Axis must be of dimension 2")

        shape = self.shape
        idx_0, idx_1 = ax
        if (not isinstance(idx_0, int) or not isinstance(idx_1, int)):
            raise ValueError("Indices must be integers")

        if shape[idx_0] != shape[idx_1]:
            raise ValueError("Indices to contract must have the same dimension")

        if not new_name:
            new_name = self.name

        # Create an identity matrix of the appropriate dimension
        dimension = shape[idx_0]
        identity_matrix = np.eye(dimension)

        # Use np.tensordot to contract the tensor with the identity matrix
        contracted_tensor = np.tensordot(self.tensor, identity_matrix, axes=(ax, [0, 1]))
        if contracted_tensor.shape == ():
            new_tensor = np.asarray([contracted_tensor])
            return Node(new_name, new_tensor, connected_nodes = self.connected_nodes)

        return Node(new_name, contracted_tensor, connected_nodes = self.connected_nodes)
