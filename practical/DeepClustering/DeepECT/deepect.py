from queue import Queue
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.utils
import torch.utils.data
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
from clustpy.deep._utils import set_torch_seed
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.utils import check_random_state
from clustpy.data.real_torchvision_data import load_mnist
from clustpy.data import load_reuters
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from tqdm import tqdm
import math


class Cluster_Node:
    """
    This class represents a cluster node within a binary cluster tree. Each node in a cluster tree represents a cluster. The cluster is
    stored through its center (self.center). During the assignment of a new minibatch to the cluster tree, each node stores the samples which are
    nearest to its center (self.assignments).
    The centers of leaf nodes are optimized through autograd, whereas the center of inner nodes are adapted analytically with weights for each of its
    child stored in self.weights.
    """

    def __init__(
        self,
        center: np.ndarray,
        device: torch.device,
        id: int = 0,
        parent: "Cluster_Node" = None,
        split_id: int = 0,
        weight: int = 1,
    ) -> "Cluster_Node":
        """
        Constructor for class Cluster_Node

        Parameters
        ----------
        center : np.ndarray
            The initial center for this node.
        device : torch.device
            The device to be trained on.
        id : int, optional
            The ID of the node, by default 0.
        parent : Cluster_Node, optional
            The parent node, by default None.

        Returns
        -------
        Cluster_Node
            The initialized Cluster_Node object.
        """
        self.device = device
        self.left_child = None
        self.right_child = None
        self.weight = torch.tensor(weight, dtype=torch.float)
        self.center = torch.nn.Parameter(
            torch.tensor(
                center, requires_grad=True, device=self.device, dtype=torch.float
            )
        )
        self.assignments: Union[torch.Tensor | None] = None
        self.assignment_indices: Union[torch.Tensor | None] = None
        self.sum_squared_dist: Union[torch.Tensor | None] = None
        self.id = id
        self.split_id = split_id
        self.parent = parent

    def clear_assignments(self):
        if self.left_child is not None:
            self.left_child.clear_assignments()
        if self.right_child is not None:
            self.right_child.clear_assignments()
        self.assignments = None
        self.assignment_indices = None
        self.sum_squared_dist = None

    def is_leaf_node(self):
        """
        Tests wether the this node is a leaf node.

        Returns
        -------
        boolean
            True if the node is a leaf node else False
        """
        return self.left_child is None and self.right_child is None

    def from_leaf_to_inner(self):
        """
        Converts a leaf node to an inner node. Weights for its child are initialised and the centers are not trainable anymore.

        """
        # inner node on cpu
        self.center.requires_grad = False
        self.assignments = None
        self.sum_squared_dist = None

    def prune(self):
        if self.left_child is not None:
            self.left_child.prune()
        if self.right_child is not None:
            self.right_child.prune()
        self.from_leaf_to_inner()

    def set_childs(
        self,
        optimizer: Union[torch.optim.Optimizer | None],
        left_child_centers: np.ndarray,
        left_child_weight: int,
        right_child_centers: np.ndarray,
        right_child_weight: int,
        max_id: int = 0,
        max_split_id: int = 0,
    ):
        """
        Set new childs to this cluster node and therefore changes this node to an inner node.

        Parameters
        ----------
        left_child_centers : np.array
            initial centers for the left child
        right_child_centers : np.array
            initial centers for the right child
        split : int
            indicates the current split count (max count of clusters)

        """
        self.left_child = Cluster_Node(
            left_child_centers,
            self.device,
            max_id + 1,
            self,
            max_split_id + 1,
            left_child_weight,
        )
        self.right_child = Cluster_Node(
            right_child_centers,
            self.device,
            max_id + 2,
            self,
            max_split_id + 1,
            right_child_weight,
        )
        self.from_leaf_to_inner()

        if optimizer is not None:
            optimizer.add_param_group({"params": self.left_child.center})
            optimizer.add_param_group({"params": self.right_child.center})


class Cluster_Tree:
    """
    This class represents a binary cluster tree. It provides multiple functionalities used for improving the cluster tree, like calculating
    the DC and NC losses for the tree and assigning samples of a minibatch to the appropriate nodes. Furthermore it provides methods for
    growing and pruning the tree as well as the analytical adaption of the inner nodes.
    """

    def __init__(
        self,
        init_leafnode_centers: np.ndarray,
        init_labels: np.ndarray,
        device: torch.device,
    ) -> "Cluster_Tree":
        """
        Constructor for class Cluster_Tree

        Parameters
        ----------
        init_leafnode_ceners : np.array
            the centers of the two initial leaf nodes of the tree given as a array of shape(2,#embedd_features)
        device : torch.device
            device to be trained on

        Returns
        -------
        Cluster_Tree object
        """
        # center of root can be a dummy-center since its never needed
        self.root = Cluster_Node(np.zeros(init_leafnode_centers.shape[1]), device)
        # assign the 2 initial leaf nodes with its initial centers
        self.root.set_childs(
            None,
            init_leafnode_centers[0],
            init_labels[0],
            init_leafnode_centers[1],
            init_labels[1],
        )

    @property
    def number_nodes(self):
        def count_recursive(node: Cluster_Node):
            if node.is_leaf_node():
                return 1
            return (
                1 + count_recursive(node.left_child) + count_recursive(node.right_child)
            )

        return count_recursive(self.root)

    @property
    def nodes(self):
        def get_nodes_recursive(node: Cluster_Node):
            result = [node]
            if node.is_leaf_node():
                return result
            result.extend(get_nodes_recursive(node.left_child))
            result.extend(get_nodes_recursive(node.right_child))
            return result

        return get_nodes_recursive(self.root)

    def clear_node_assignments(self):
        self.root.clear_assignments()

    def get_all_leaf_nodes(self) -> List[Cluster_Node]:
        """
        Returns a list of all leaf node references of this tree

         Returns
         -------
         List[Cluster_Node]
             References of all leaf nodes in the tree
        """
        leafnodes = []
        self._collect_leafnodes(self.root, leafnodes)
        return leafnodes

    def _collect_leafnodes(self, node: Cluster_Node, leafnodes: list):
        """
        Helper function for recursively collecting all leaf nodes

        Parameters
        ----------
        node : Cluster_Node
            The node where the search for leaf nodes begin
        leafnodes : list
            This list stores the leaf nodes founded by traversing the tree
        """
        if node.is_leaf_node():
            leafnodes.append(node)
        else:
            self._collect_leafnodes(node.left_child, leafnodes)
            self._collect_leafnodes(node.right_child, leafnodes)

    def get_all_result_nodes(self, number_classes: int) -> List[Cluster_Node]:
        """
        Returns a list of all class node references for the given number of classes. This nodes
        are the leaf nodes in the tree cut for the given number of classes. The number of returned nodes
        is equal to the given number of classes.

        Parameters
        ----------
        number_classes : int
            The number of clusters which should be obtained from the cluster tree.

        Returns
        -------
        List[Cluster_Node]
            References of all nodes representing the given number of clusters.
        """
        result_nodes = []
        max_split_level = sorted(list(set([node.split_id for node in self.nodes])))[
            number_classes - 1
        ]

        # the leaf nodes after the first <number_classes> - 1 growing steps (splits) are the nodes representing the <number_classes> clusters
        def get_nodes_at_split_level(node: Cluster_Node):
            if (
                node.is_leaf_node() or node.left_child.split_id > max_split_level
            ) and node.split_id <= max_split_level:
                result_nodes.append(node)
                return
            get_nodes_at_split_level(node.left_child)
            get_nodes_at_split_level(node.right_child)

        get_nodes_at_split_level(self.root)
        # consistency check
        assert (
            len(result_nodes) == number_classes
        ), "Number of cluster nodes doesn't correspond to number of classes"
        return result_nodes

    def _collect_all_result_nodes(
        self, node: Cluster_Node, split_level: int, result_nodes: list
    ):
        """
        Helper function for recursively collecting all leaf nodes of the cluster tree cut at split level <split_level>.

        Parameters
        ----------
        node : Cluster_Node
            The node where the search for result nodes starts
        split_level: int
            The split level where the tree should be cuted. A split level i of a node n states that node n
            was created in the i-th tree grow step (i-th split).
        result_nodes : list
            This list contains all leaf nodes of tree cut at level <split_level>.
        """
        # a node is part of the result if it is a leaf node of the cluster tree cut at split level <split_level>
        if node.is_leaf_node() or node.split_id > split_level:
            result_nodes.append(node)
        else:
            self._collect_all_result_nodes(node.left_child, split_level, result_nodes)
            self._collect_all_result_nodes(node.right_child, split_level, result_nodes)

    def assign_to_nodes(
        self, minibatch_embedded: torch.tensor, compute_sum_dist: bool = False
    ):
        """
        This method assigns all samples in the minibatch to its nearest nodes in the cluster tree. It is performed bottom up, so each
        sample is first assigned to its nearest leaf node. Afterwards the samples are assigned recursivley to the inner nodes by merging
        the assignments of the child node.

        Parameters
        ----------
        autoencoder: torch.nn.Module
            Autoencoder used to calculate the embeddings
        minibatch : torch.Tensor
            The minibatch with shape (#samples, #emb_features)
        """
        # retrieve all leaf nodes from the current tree
        leafnodes = self.get_all_leaf_nodes()
        # transform it into a list of leaf node centers and stack it into a tensor of shape (#leafnodes, #emb_features)
        leafnode_centers = list(map(lambda node: node.center.data, leafnodes))
        leafnode_tensor = torch.stack(leafnode_centers, dim=0)  # (k, d)

        #  calculate the distance from each sample in the minibatch to all leaf nodes
        with torch.no_grad():
            distance_matrix = torch.cdist(
                minibatch_embedded, leafnode_tensor, p=2
            )  # kmeans uses L_2 norm (euclidean distance) (b, k)
        # the sample gets the nearest node assigned
        min_dists, assignments = torch.min(distance_matrix, dim=1)

        # for each leafnode, check which samples it has got assigned and store the assigned samples in the leafnode
        for i, node in enumerate(leafnodes):
            indices = (assignments == i).nonzero()
            if len(indices) < 1:
                node.assignments = (
                    None  # store None (perhaps overwrite previous assignment)
                )
                node.assignment_indices = None
                node.sum_squared_dist = None
            else:
                leafnode_data = minibatch_embedded[indices.squeeze()]
                if leafnode_data.ndim == 1:
                    leafnode_data = leafnode_data[None]
                node.assignments = leafnode_data
                node.assignment_indices = indices.reshape(
                    indices.nelement()
                )  # one dimensional tensor containing the indices
                if compute_sum_dist:
                    node.sum_squared_dist = torch.sum(
                        min_dists[indices.squeeze()].pow(2)
                    )

    def _assign_to_splitnodes(self, node: Cluster_Node):
        """
        Function for recursively assigning samples to inner nodes by merging the assignments of its two childs

        Parameters
        ----------
        node : Cluster_Node
            The node where the assignmets should be stored
        """
        if node.is_leaf_node():  # base case of recursion
            return node.assignment_indices, node.assignments
        else:
            # get assignments of left child
            left_assignment_indices, left_assignments = self._assign_to_splitnodes(
                node.left_child
            )
            # get assignments of right child
            right_assignment_indices, right_assignments = self._assign_to_splitnodes(
                node.right_child
            )
            # if one of the assignments is empty, then just use the assignments of the other node
            if left_assignments == None or right_assignments == None:
                node.assignments = (
                    left_assignments if right_assignments == None else right_assignments
                )
                node.assignment_indices = (
                    left_assignment_indices
                    if right_assignments == None
                    else right_assignment_indices
                )
            else:
                # merge the assignments of the child nodes and store it in the nodes
                node.assignments = torch.cat(
                    (left_assignments, right_assignments), dim=0
                )
                node.assignment_indices = torch.cat(
                    (left_assignment_indices, right_assignment_indices), dim=0
                )
            return node.assignment_indices, node.assignments

    def nc_loss(self) -> torch.Tensor:
        """
        Function for calculating the nc loss used for adopting the leaf node centers.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            The autoencoder used for calculating the embeddings

        Returns
        -------
        loss : torch.Tensor
            the NC loss
        """
        leaf_nodes = self.get_all_leaf_nodes()
        # convert the list of leaf nodes to a list of the corresponding leaf node centers as tensors
        leafnode_centers = [
            node.center for node in leaf_nodes if node.assignments is not None
        ]
        if len(leafnode_centers) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=leaf_nodes[0].device)
        # !!! Maybe here a problem of concatenating parameter tensors !!
        # reformat list of tensors to one sinlge tensor of shape (#leafnodes,#emb_features)
        leafnode_center_tensor = torch.stack(leafnode_centers, dim=0)

        # calculate the center of the assignments from the current minibatch for each leaf node
        with torch.no_grad():  # embedded space should not be optimized in this loss
            # get the assignments for each leaf node (from the current minibatch)
            leafnode_assignments = [
                node.assignments for node in leaf_nodes if node.assignments is not None
            ]
            leafnode_minibatch_centers = list(
                map(
                    lambda assignments: torch.sum(assignments, axis=0)
                    / len(assignments),
                    leafnode_assignments,
                )
            )
        # reformat list of tensors to one sinlge tensor of shape (#leafnodes,#emb_features)
        leafnode_minibatch_centers_tensor = torch.stack(
            leafnode_minibatch_centers, dim=0
        )

        # calculate the distance between the current leaf node centers and the center of its assigned embeddings averaged over all leaf nodes
        distance = torch.sum(
            (leafnode_center_tensor - leafnode_minibatch_centers_tensor) ** 2, axis=1
        )
        distance = torch.sqrt(distance)
        loss = torch.sum(distance) / len(leafnode_center_tensor)
        return loss

    def dc_loss(self, batchsize: int) -> torch.Tensor:
        """
        Function for calculating the overall dc loss used for improving the embedded space for a better clustering result.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            The autoencoder used for calculating the embeddings
        batchsize   : int
            The batch size used for normalization here

        Returns
        -------
        loss : torch.Tensor
            the DC loss
        """
        # batchsize = self.root.assignments.size(dim=0)
        sibling_losses = []  # storing losses for each node in tree
        self._calculate_sibling_loss(self.root, sibling_losses)
        number_nodes = self.number_nodes - 1  # exclude root node
        # make sure that each node got a loss
        assert number_nodes == len(sibling_losses)

        # transform list of losses for each node to a tensor
        sibling_losses = torch.stack(sibling_losses, dim=0)
        # calculate overall dc loss
        loss = torch.sum(sibling_losses) / (number_nodes * batchsize)
        return loss

    def _calculate_sibling_loss(
        self,
        root: Cluster_Node,
        sibling_loss: List[torch.Tensor],
    ) -> int:
        """
        Helper function for recursively calculating the dc loss for each node. The losses are stored in the given list <sibling_loss>

        Parameters
        ----------
        root : Cluster_Node
            The node for which childs (siblings) the dc loss should be calculated
        autoencoder : torch.nn.Module
            The autoencoder used for calculating the embeddings
        sibling_loss : List[torch.Tensor]
            Stores the loss for each node

        Returns
        -------
        #nodes : int
            Returns the number of nodes in the cluster tree
        """
        if root is None:
            return

        # Traverse the left subtree
        self._calculate_sibling_loss(root.left_child, sibling_loss)

        # Traverse the right subtree
        self._calculate_sibling_loss(root.right_child, sibling_loss)

        # Calculate lc loss for siblings if they exist
        if root.left_child and root.right_child:
            # calculate dc loss for left child with respect to the right child
            loss_left = self._single_sibling_loss(root.left_child, root.right_child)
            # calculate dc loss for right child with respect to the left child
            loss_right = self._single_sibling_loss(root.right_child, root.left_child)
            # store the losses
            sibling_loss.extend([loss_left, loss_right])

    def _single_sibling_loss(
        self, node: Cluster_Node, sibling: Cluster_Node
    ) -> torch.Tensor:
        """
        Calculates a single dc loss for the node <node> with respect to its sibling <sibling>.

        Parameters
        ----------
        node : Cluster_Node
            The node for which the dc loss should be calculated
        sibling : Cluster_Node
            The sibling of the node for which the dc loss should be calculated
        autoencoder : torch.nn.Module
            The autoencoder used for calculating the embeddings

        Returns
        -------
        loss : torch.Tensor
            the dc loss for <node>
        """
        if node.assignments is None:
            return torch.tensor(0.0, dtype=torch.float, device=node.device)
        # calculate direction (norm) vector between <node> and <sibling>
        sibling_direction = (
            node.center.detach() - sibling.center.detach()
        ) / torch.sqrt(torch.sum((node.center.detach() - sibling.center.detach()) ** 2))
        # transform tensor from 1d to 2d
        sibling_direction = sibling_direction[None]
        # project each sample assigned to <node> in the direction of its sibling and sum up the absolute projection values for each sample
        loss = torch.sum(
            torch.abs(
                torch.matmul(
                    sibling_direction,
                    -(node.assignments - node.center.detach()).T,
                )
            )
        )
        return loss

    def adapt_inner_nodes(self, root: Cluster_Node):
        """
        Function for recursively assigning samples to inner nodes by merging the assignments of its two childs

        Parameters
        ----------
        node : Cluster_Node
            The node where the assignmets should be stored
        """
        if root is None:
            return

        # Traverse the left subtree
        self.adapt_inner_nodes(root.left_child)

        # Traverse the right subtree
        self.adapt_inner_nodes(root.right_child)

        # adapt node based on this 2 childs
        if root.left_child and root.right_child:
            # adapt weight for left child
            left_child_len_assignments = len(
                root.left_child.assignments
                if root.left_child.assignments is not None
                else []
            )
            root.left_child.weight = 0.5 * (
                root.left_child.weight + left_child_len_assignments
            )
            # adapt weight for right child
            right_child_len_assignments = len(
                root.right_child.assignments
                if root.right_child.assignments is not None
                else []
            )
            root.right_child.weight = 0.5 * (
                root.right_child.weight + right_child_len_assignments
            )
            # adapt center of parent based on the new weights
            with torch.no_grad():
                child_centers = torch.stack(
                    (
                        root.left_child.weight * root.left_child.center,
                        root.right_child.weight * root.right_child.center,
                    ),
                    dim=0,
                )
                root.center = torch.sum(child_centers, axis=0) / torch.add(
                    root.left_child.weight, root.right_child.weight
                )
            root.assignments = torch.zeros(
                left_child_len_assignments + right_child_len_assignments,
                dtype=torch.int8,
                device=root.device,
            )

    def prune_tree(self, pruning_threshold: float):
        """
        Prunes the tree by removing nodes with weights below the pruning threshold.

        Args:
            pruning_threshold (float): The threshold value for pruning. Nodes with weights below this threshold will be removed.

        Returns:
            None
        """

        def prune_node(parent: Cluster_Node, child_attr: str):
            """
            Prunes a node from the tree by replacing it with its child or sibling node.

            Args:
                parent (Cluster_Node): The parent node from which the child or sibling node will be pruned.
                child_attr (str): The attribute name of the child node to be pruned.

            Returns:
                None
            """
            child_node: Cluster_Node = getattr(parent, child_attr)
            sibling_attr = (
                "left_child" if child_attr == "right_child" else "right_child"
            )
            sibling_node: Cluster_Node = getattr(parent, sibling_attr)

            if sibling_node is None:
                # if parent == self.root:
                #     self.root = child_node
                #     self.root.parent = None
                # else:
                #     grandparent = parent.parent
                #     if grandparent.left_child == parent:
                #         grandparent.left_child = child_node
                #     else:
                #         grandparent.right_child = child_node
                #     if child_node is not None:
                #         child_node.parent = grandparent
                raise ValueError(sibling_node)
            else:
                if parent == self.root:
                    self.root = sibling_node
                    self.root.parent = None
                else:
                    grandparent = parent.parent
                    if grandparent.left_child == parent:
                        grandparent.left_child = sibling_node
                    else:
                        grandparent.right_child = sibling_node
                    sibling_node.parent = grandparent
                sibling_node.split_id = parent.split_id
                child_node.prune()
                del child_node
                del parent
                print(
                    f"Tree size after pruning: {self.number_nodes}, leaf nodes: {len(self.get_all_leaf_nodes())}"
                )

        def prune_recursive(node: Cluster_Node):
            """
            Recursively prunes the tree starting from the given node.

            Parameters:
            node (Cluster_Node): The node from which to start pruning.

            Returns:
            None
            """
            if node.left_child:
                prune_recursive(node.left_child)
            if node.right_child:
                prune_recursive(node.right_child)

            if node.weight < pruning_threshold:
                if node.parent is not None:
                    if node.parent.left_child == node:
                        prune_node(node.parent, "left_child")
                    else:
                        prune_node(node.parent, "right_child")
                else:
                    if (
                        self.root.left_child
                        and self.root.left_child.weight < pruning_threshold
                    ):
                        prune_node(self.root, "left_child")
                    if (
                        self.root.right_child
                        and self.root.right_child.weight < pruning_threshold
                    ):
                        prune_node(self.root, "right_child")

        prune_recursive(self.root)

    def grow_tree(
        self,
        dataloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        seed: int,
        device: Union[torch.device | str],
    ) -> None:
        """Grows the tree at the leaf node with the highest squared distance between its assignments and center.
        The distance is not normalized, so larger clusters will be chosen.

        We transform the dataset (or a representative sub-sample of it)
        onto the embedded space. Then, we determine the leaf node
        with the highest sum of squared distances between its center
        and the assigned data points. We selected this rule because it
        provides a good balance between the number of data points
        and data variance for this cluster.
        Next, we split the selected node and attach two new leaf
        nodes to it as children. We determine the initial centers for
        these new leaf nodes by applying two-means (k-means with
        k = 2) to the assigned data points.
        """
        leaf_nodes = self.get_all_leaf_nodes()
        batched_seqential_loader = torch.utils.data.DataLoader(
            dataloader.dataset, dataloader.batch_size, shuffle=False
        )
        with torch.no_grad():
            leaf_node_dist_sums = torch.zeros(
                len(leaf_nodes), dtype=torch.float, device="cpu"
            )
            for batch in batched_seqential_loader:
                idxs, x = batch
                embed = autoencoder.encode(x.to(device))
                self.assign_to_nodes(embed, compute_sum_dist=True)
                leaf_node_dist_sums += torch.stack(
                    [
                        (
                            leaf.sum_squared_dist.cpu()
                            if leaf.sum_squared_dist is not None
                            else torch.tensor(0, dtype=torch.float32, device="cpu")
                        )
                        for leaf in leaf_nodes
                    ],
                    dim=0,
                )

            idx = leaf_node_dist_sums.argmax()
            highest_dist_leaf_node = leaf_nodes[idx]
            # get all assignments for highest dist leaf node
            assignments = []
            for batch in batched_seqential_loader:
                idxs, x = batch
                embed = autoencoder.encode(x.to(device))
                self.assign_to_nodes(embed)
                if highest_dist_leaf_node.assignments is not None:
                    assignments.append(highest_dist_leaf_node.assignments.cpu())
            child_assignments = KMeans(
                n_clusters=2, n_init="auto", random_state=seed
            ).fit(torch.cat(assignments, dim=0).numpy())
            print(f"Leaf assignments: {len(child_assignments.labels_)}")
            child_weights = np.array(
                [
                    len(child_assignments.labels_[child_assignments.labels_ == i])
                    for i in np.unique(child_assignments.labels_)
                    if i >= 0
                ]
            )
            highest_dist_leaf_node.set_childs(
                optimizer,
                child_assignments.cluster_centers_[0],
                child_weights[0],
                child_assignments.cluster_centers_[1],
                child_weights[1],
                max([leaf.id for leaf in leaf_nodes]),
                max([node.split_id for node in self.nodes]),
            )
            print(
                f"Tree size after growing: {self.number_nodes}, leaf nodes: {len(self.get_all_leaf_nodes())}"
            )


class _DeepECT_Module(torch.nn.Module):
    """
    The _DeepECT_Module. Contains most of the algorithm specific procedures like the loss and tree-grow functions.


    Parameters
    ----------
    init_centers : np.ndarray
        The initial cluster centers
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    cluster_tree: Cluster_Node
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(
        self,
        init_leafnode_centers: np.ndarray,
        init_labels: np.ndarray,
        device: torch.device,
        seed: int,
        augmentation_invariance: bool = False,
    ):
        super().__init__()
        self.augmentation_invariance = augmentation_invariance
        # Create initial cluster tree
        self.cluster_tree = Cluster_Tree(
            init_leafnode_centers,
            np.array(
                [
                    len(init_labels[init_labels == i])
                    for i in np.unique(init_labels)
                    if i >= 0
                ]
            ),
            device,
        )
        self.device = device
        self.random_state = seed

    def deepect_augmentation_invariance_loss(
        self, embedded: torch.Tensor, embedded_aug: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """
        Calculate the DeepECT loss of given embedded samples with augmentation invariance.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        embedded_aug : torch.Tensor
            the embedded augmented samples
        alpha : float
            the alpha value

        Returns
        -------
        loss : torch.Tensor
            the final DeepECT loss
        """
        # # Get loss of non-augmented data
        # squared_diffs = squared_euclidean_distance(embedded, self.centers)
        # probs = _DeepECT_get_probs(squared_diffs, alpha)
        # clean_loss = (squared_diffs.sqrt() * probs).sum(1).mean()
        # # Get loss of augmented data
        # squared_diffs_augmented = squared_euclidean_distance(embedded_aug, self.centers)
        # aug_loss = (squared_diffs_augmented.sqrt() * probs).sum(1).mean()
        # # average losses
        # loss = (clean_loss + aug_loss) / 2
        loss = None
        return loss

    def fit(
        self,
        autoencoder: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        max_iterations: int,
        pruning_threshold: float,
        grow_interval: int,
        max_leaf_nodes: int,
        optimizer: torch.optim.Optimizer,
        rec_loss_fn: torch.nn.modules.loss._Loss,
        device: Union[torch.device | str],
    ) -> "_DeepECT_Module":
        """
        Trains the _DeepECT_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            the autoencoder
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        max_iteratins : int
            number of iterations for the clustering procedure.
        optimizer : torch.optim.Optimizer
            the optimizer for training
        rec_loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss

        Returns
        -------
        self : _DeepECT_Module
            this instance of the _DeepECT_Module
        """

        train_iterator = iter(trainloader)

        for e in tqdm(range(max_iterations), desc="Fit", total=max_iterations):
            self.cluster_tree.prune_tree(pruning_threshold)
            if (e > 0 and e % grow_interval == 0) or self.cluster_tree.number_nodes < 3:
                if len(self.cluster_tree.get_all_leaf_nodes()) >= max_leaf_nodes:
                    break
                self.cluster_tree.grow_tree(
                    trainloader,
                    autoencoder,
                    optimizer,
                    seed=self.random_state,
                    device=device,
                )

            # retrieve minibatch (endless)
            try:
                # get next minibatch
                batch = next(train_iterator)
            except StopIteration:
                # after full epoch shuffle again
                train_iterator = iter(trainloader)
                batch = next(train_iterator)
            idxs, M = batch
            # assign data points to leafnodes and splitnodes
            rec_loss, embedded, reconstructed = autoencoder.loss(
                batch, rec_loss_fn, self.device
            )

            self.cluster_tree.assign_to_nodes(embedded)

            # calculate cluster loss
            nc_loss = self.cluster_tree.nc_loss()
            dc_loss = self.cluster_tree.dc_loss(len(M))

            loss = nc_loss + dc_loss + rec_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # adapt centers of split nodes analytically
            self.cluster_tree.adapt_inner_nodes(self.cluster_tree.root)
            self.cluster_tree.clear_node_assignments()
        return self

    def predict(
        self,
        number_classes: int,
        dataloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
    ) -> Tuple[np.array, np.array]:
        """
        Batchwise prediction of the given samples in the dataloader for a given number of classes.

        Parameters
        ----------
        number_classes : torch.Tensor
            Number of clusters which should be identified
        dataloader : torch.utils.data.DataLoader
            Contains the samples which should be predicted
        autoencoder: torch.nn.Module
            Used to calculate the embeddings

        Returns
        -------
        predictions_numpy : numpy.array
            The predicted labels
        centers: numpy.array
            The centers of the <numb_classes> clusters
        """

        # get all resulting leaf nodes after cutting the tree for <number_classes> clusters
        cluster_nodes = self.cluster_tree.get_all_result_nodes(number_classes)

        with torch.no_grad():
            # perform prediction batchwise
            predictions = []
            for batch in tqdm(dataloader, desc="Predict"):
                # calculate embeddings of the samples which should be predicted
                batch_data = batch[1].to(self.device)
                embeddings = autoencoder.encode(batch_data)
                # assign the embeddings to the cluster tree
                self.cluster_tree.assign_to_nodes(embeddings)
                self.cluster_tree._assign_to_splitnodes(self.cluster_tree.root)
                # retrieve a list which stores the assigned sample indices for each node in <cluster_nodes>
                assignments_list = list(
                    map(
                        lambda node: (
                            node.assignment_indices
                            if node.assignment_indices is not None
                            else torch.tensor([], dtype=torch.int, device=self.device)
                        ),
                        cluster_nodes,
                    )
                )
                # create a 1-dim. tensor which stores the labels for the samples
                labels = torch.cat(
                    [
                        torch.full(
                            (len(assignments) if assignments is not None else 0,),
                            i,
                            dtype=torch.int,
                        )
                        for i, assignments in enumerate(assignments_list)
                    ]
                ).cpu()
                # flatten the list of assignments for each node to a single 1-dim. tensor
                assignments = torch.cat(assignments_list)
                # sort the labels based on the ordering of the samples in the batch
                sorted_assignment_indices = torch.argsort(assignments).cpu()
                predictions.append(labels[sorted_assignment_indices])
                self.cluster_tree.clear_node_assignments()

            # transform the predictions for each batch to a single output array for the whole dataset
            predictions_numpy = torch.cat(predictions, dim=0).numpy()

            # return corresponding centers as well
            centers = list(map(lambda node: node.center.data, cluster_nodes))
            # reformat list of tensors to one sinlge tensor of shape (#leafnodes,#emb_features)
            centers = torch.stack(centers, dim=0).cpu().numpy()

        return predictions_numpy, centers


def _deep_ect(
    X: np.ndarray,
    batch_size: int,
    pretrain_optimizer_params: dict,
    clustering_optimizer_params: dict,
    pretrain_epochs: int,
    number_classes: int,
    max_iterations: int,
    pruning_threshold: float,
    grow_interval: int,
    optimizer_class: torch.optim.Optimizer,
    rec_loss_fn: torch.nn.modules.loss._Loss,
    autoencoder: torch.nn.Module,
    embedding_size: int,
    max_leaf_nodes: int,
    custom_dataloaders: tuple,
    augmentation_invariance: bool,
    seed: int,
    autoencoder_save_param_path: str = "pretrained_ae.pth",
):
    """
    Start the actual DeepECT clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    max_iterations : int
        number of iterations for the actual clustering procedure.
    pruning_treshold : float
        the treshold for pruning the tree
    optimizer_class : torch.optim.Optimizer
        the optimizer
    rec_loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DeepECT after the training terminated,
        The cluster centers as identified by DeepECT after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    save_ae_state_dict = not hasattr(autoencoder, "fitted") or not autoencoder.fitted
    (
        device,
        trainloader,
        testloader,
        autoencoder,
        _,
        n_clusters,
        init_labels,
        init_leafnode_centers,
        _,
    ) = get_standard_initial_deep_clustering_setting(
        X,
        2,
        batch_size,
        pretrain_optimizer_params,
        pretrain_epochs,
        optimizer_class,
        rec_loss_fn,
        autoencoder,
        embedding_size,
        custom_dataloaders,
        KMeans,
        {"random_state": seed},
        seed,
    )
    print(device)
    if save_ae_state_dict:
        autoencoder.save_parameters(autoencoder_save_param_path)
    # Setup DeepECT Module
    deepect_module = _DeepECT_Module(
        init_leafnode_centers, init_labels, device, seed, augmentation_invariance
    ).to(device)
    # Use DeepECT optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(
        list(autoencoder.parameters()), **clustering_optimizer_params
    )
    # DeepECT Training loop
    deepect_module.fit(
        autoencoder,
        trainloader,
        max_iterations,
        pruning_threshold,
        grow_interval,
        max_leaf_nodes,
        optimizer,
        rec_loss_fn,
        device,
    )
    # Get labels
    DeepECT_labels, DeepECT_centers = deepect_module.predict(
        number_classes, testloader, autoencoder
    )
    return deepect_module.cluster_tree, DeepECT_labels, DeepECT_centers, autoencoder


class DeepECT:

    def __init__(
        self,
        batch_size: int = 256,
        pretrain_optimizer_params: dict = None,
        clustering_optimizer_params: dict = None,
        pretrain_epochs: int = 50,
        number_classes: int = 2,
        max_iterations: int = 10000,
        grow_interval: int = 500,
        pruning_threshold: float = 0.1,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        rec_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        autoencoder: torch.nn.Module = None,
        embedding_size: int = 10,
        max_leaf_nodes: int = 20,
        custom_dataloaders: tuple = None,
        augmentation_invariance: bool = False,
        seed: int = 42,
        autoencoder_param_path: str = None,
    ):
        """
        The Deep Embedded Cluster Tree (DeepECT) algorithm.
        First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
        Afterward, a cluter tree will be grown and the AE will be optimized using the DeepECT loss function.

        Parameters
        ----------
        batch_size : int
            size of the data batches (default: 256)
        pretrain_optimizer_params : dict
            parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
        clustering_optimizer_params : dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
        pretrain_epochs : int
            number of epochs for the pretraining of the autoencoder (default: 50)
        max_iterations : int
            number of iteratins for the actual clustering procedure (default: 1000)
        optimizer_class : torch.optim.Optimizer
            the optimizer class (default: torch.optim.Adam)
        rec_loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction (default: torch.nn.MSELoss())
        autoencoder : torch.nn.Module
            the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
        embedding_size : int
            size of the embedding within the autoencoder (default: 10)
        custom_dataloaders : tuple
            tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
            If None, the default dataloaders will be used (default: None)
        augmentation_invariance : bool
            If True, augmented samples provided in custom_dataloaders[0] will be used to learn
            cluster assignments that are invariant to the augmentation transformations (default: False)
        random_state : np.random.RandomState
            use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

        Attributes
        ----------
        labels_ : np.ndarray
            The final labels (obtained by a final KMeans execution)
        cluster_centers_ : np.ndarray
            The final cluster centers (obtained by a final KMeans execution)
        DeepECT_labels_ : np.ndarray
            The final DeepECT labels
        DeepECT_cluster_centers_ : np.ndarray
            The final DeepECT cluster centers
        autoencoder : torch.nn.Module
            The final autoencoder
        """
        if max_leaf_nodes < number_classes:
            raise ValueError(
                f"The given maximal number of leaf nodes ({max_leaf_nodes}) is smaller than the given number of classes ({number_classes})"
            )
        self.batch_size = batch_size
        self.pretrain_optimizer_params = (
            {"lr": 1e-3}
            if pretrain_optimizer_params is None
            else pretrain_optimizer_params
        )
        self.clustering_optimizer_params = (
            {"lr": 1e-4}
            if clustering_optimizer_params is None
            else clustering_optimizer_params
        )
        self.pretrain_epochs = pretrain_epochs
        self.number_classes = number_classes
        self.max_iterations = max_iterations
        self.grow_interval = grow_interval
        self.pruning_threshold = pruning_threshold
        self.optimizer_class = optimizer_class
        self.rec_loss_fn = rec_loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.max_leaf_nodes = max_leaf_nodes
        self.seed = seed
        self.autoencoder_param_path = autoencoder_param_path
        set_torch_seed(self.seed)

    def fit(self, X: np.ndarray) -> "DeepECT":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set as a 2d-array of shape (#samples, #features)

        Returns
        -------
        self : DeepECT
            this instance of the DeepECT algorithm
        """
        # augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)

        DeepECT_tree, DeepECT_labels, DeepECT_centers, autoencoder = _deep_ect(
            X,
            self.batch_size,
            self.pretrain_optimizer_params,
            self.clustering_optimizer_params,
            self.pretrain_epochs,
            self.number_classes,
            self.max_iterations,
            self.pruning_threshold,
            self.grow_interval,
            self.optimizer_class,
            self.rec_loss_fn,
            self.autoencoder,
            self.embedding_size,
            self.max_leaf_nodes,
            self.custom_dataloaders,
            self.augmentation_invariance,
            self.seed,
            self.autoencoder_param_path,
        )
        self.DeepECT_labels_ = DeepECT_labels
        self.DeepECT_cluster_centers_ = DeepECT_centers
        self.DeepECT_tree_ = DeepECT_tree
        self.autoencoder = autoencoder
        return self


if __name__ == "__main__":
    # dataset, labels = load_mnist(return_X_y=True)
    # autoencoder = FeedforwardAutoencoder([dataset.shape[1], 500, 500, 2000, 10])
    # autoencoder.load_state_dict(
    #     torch.load("practical/DeepClustering/DeepECT/pretrained_AE.pth")
    # )
    # autoencoder.fitted = True
    # deepect = DeepECT(number_classes=10, autoencoder=autoencoder, max_leaf_nodes=20)
    # deepect.fit(dataset)
    # print(unsupervised_clustering_accuracy(labels, deepect.DeepECT_labels_))

    dataset, labels = load_reuters(return_X_y=True)
    print(dataset.shape[0])
    autoencoder = FeedforwardAutoencoder(
        [dataset.shape[1], 500, 500, 2000, 10],
    )
    autoencoder.load_parameters(
        "practical/DeepClustering/DeepECT/pretrained_AE_reuters.pth"
    )
    autoencoder.fitted = True
    deepect = DeepECT(
        max_iterations=50000,
        number_classes=4,
        embedding_size=10,
        pretrain_epochs=19,
        max_leaf_nodes=12,
        autoencoder=autoencoder,
        autoencoder_param_path="practical/DeepClustering/DeepECT/pretrained_AE_reuters.pth",
    )
    deepect.fit(dataset)
    print(unsupervised_clustering_accuracy(labels, deepect.DeepECT_labels_))
    torch.save(deepect, "practical/DeepClustering/DeepECT/reuters_deepect.pth")
