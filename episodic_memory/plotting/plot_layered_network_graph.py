#!/usr/bin/env python
"""
Plot multi-graphs in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

pause = False


class LayeredNetworkGraph(object):
    def __init__(self, graphs, interaction_matrices=None, node_labels=None, layout=nx.spring_layout, ax=None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network and interaction matrices
        [(xi_{g1 g2}, xi_{g2 g2}, <symmetric?>),... ],
        plot the network in 3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Parameters:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        interaction_matrices: list of tuples of interaction matrices between nodes in different layers
            The parameter has the form [ (xi_{g1 g2}, xi_{g2 g1}, <symmetric?>), ... ].
            xi_{gi gj} is None if no interaction
            xi_{gi gi+1} is None if interactions are symmetric

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.
        """

        # book-keeping
        self.graphs = graphs
        self.interaction_matrices = interaction_matrices

        if interaction_matrices is None:  # Handle the case of no interaction matrices
            self.interaction_matrices = []
            for i in range(len(self.graphs) - 1):
                self.interaction_matrices.append((None, None, True))

        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        # self.draw()


    def update_node_values(self, updated):
        """
        Updates the node values of the internal node representation

        Parameters
        ----------
        updated : list
            A list of node values to be updated. Each element in the list is the value associated with each node
            So if a layer `i` has `n` nodes, `updated[i]` will have `n` elements
        """
        for layer_id in range(len(updated)):
            updated_values = updated[layer_id]

            for node_idx, node in enumerate(self.nodes):
                if node[1] == layer_id:
                    self.nodes[node_idx] = (node[0], node[1], updated_values[node[0]])


    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            for node, node_data in g.nodes(data=True):
                node_value = node_data['value'] if 'value' in node_data.keys() else 1
                self.nodes.append((node, z, node_value))
            # self.nodes.extend([(node, z) for node in g.nodes()])

    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            for source, target, edge_data in g.edges.data():
                weight = edge_data['weight'] if 'weight' in edge_data.keys() else 0
                self.edges_within_layers.append(((source, z), (target, z), weight, False))
            # self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges.data()])

    def get_edges_between_layers(self):
        """Use the interaction matrices to create edges between layers."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]

            forward_interaction, backward_interaction, symmetric = self.interaction_matrices[z1]

            # if interactions are symmetric, backward interaction is equal to forward interaction
            # if symmetric and forward_interaction is not None:
            #     backward_interaction = forward_interaction.T

            if forward_interaction is not None:
                for nodei in range(forward_interaction.shape[0]):
                    for nodej in range(forward_interaction.shape[1]):
                        self.edges_between_layers.append(((nodei, z1), (nodej, z2), forward_interaction[nodei, nodej],
                                                          not symmetric))

            if backward_interaction is not None and not symmetric:
                for nodei in range(backward_interaction.shape[0]):
                    for nodej in range(backward_interaction.shape[1]):
                        self.edges_between_layers.append(((nodei, z2), (nodej, z1), backward_interaction[nodei, nodej],
                                                          not symmetric))

    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        # What we would like to do, is apply the layout function to a combined, layered network.
        # However, networkx layout functions are not implemented for the multi-dimensional case.
        # Futhermore, even if there was such a layout function, there probably would be no straightforward way to
        # specify the planarity requirement for nodes within a layer.
        # Therefor, we compute the layout for the full network in 2D, and then apply the
        # positions to the nodes in all planes.
        # For a force-directed layout, this will approximately do the right thing.

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z): (*pos[node], z) for node in g.nodes()})

    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, *args, **kwargs):
        segments = []
        line_widths = []
        for source, target, weight, direction in edges:
            segments.append((self.node_positions[source], self.node_positions[target]))
            line_widths.append(weight)

        line_collection = Line3DCollection(segments, linewidths=line_widths, *args, **kwargs)
        self.ax.add_collection3d(line_collection)

    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
               (ymin - pad * dy, ymax + pad * dy)

    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u, v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)

    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z, node_value in self.nodes:
            if node in node_labels:
                self.ax.text(*self.node_positions[(node, z)],
                             "{}: {:.2f}".format(node_labels[node], node_value),
                             *args, **kwargs)

    def draw(self, frame_id=-1, interlayer_edges=True, node_count_limits=None, intralayer_edges=True):
        """
        Function to draw the current state of the network

        Parameters
        ----------
        frame_id : int
            The current frame
        interlayer_edges : bool
            Indicates if the interlayer edges should be plotted (default: True)
        node_count_limits : int
            The limit enforced on the number of nodes plotted. (default: None)
        intralayer_edges : bool
            Indicates if the intralayer edges should be plotted (default: True)
        """
        if node_count_limits is None:
            node_count_limits = []
            for layer_id in range(self.total_layers):
                node_count_limits.append(np.sum([1 for node in self.nodes if node[1] == layer_id]))

        self.ax.clear()
        if frame_id != -1:
            self.ax.set_title("Frame %d" % (frame_id + 1), fontweight="bold")

        if intralayer_edges:
            self.draw_edges([edge for edge in self.edges_within_layers \
                             if edge[0][0] < node_count_limits[edge[0][1]] and edge[1][0] < node_count_limits[edge[1][1]]],
                            color='k', alpha=0.3, linestyle='-', zorder=2)

        if interlayer_edges:
            self.draw_edges([edge for edge in self.edges_between_layers \
                             if
                             edge[0][0] < node_count_limits[edge[0][1]] and edge[1][0] < node_count_limits[edge[1][1]]],
                            color='k', alpha=0.3, linestyle='--', zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=1)
            self.draw_nodes([node[:2] for node in self.nodes if node[1] == z and node[0] < node_count_limits[z]],
                            c=[node[2] for node in self.nodes if node[1] == z and node[0] < node_count_limits[z]],
                            cmap="inferno",
                            s=300, zorder=3)

        if self.node_labels:
            self.draw_node_labels({key: val for key, val in self.node_labels.items() if key < node_count_limits[0]},
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)
        self.ax.set_axis_off()


def main():
    # define graphs
    n = 5
    g = nx.erdos_renyi_graph(4 * n, p=0.1)
    h = nx.erdos_renyi_graph(3 * n, p=0.2)
    i = nx.erdos_renyi_graph(2 * n, p=0.4)

    node_labels = {nn: str(nn) for nn in range(4 * n)}

    interaction_matrices = [
        (np.ones((4 * n, 3 * n)), None, True),
        (np.ones((3 * n, 2 * n)), None, True)
    ]

    # initialise figure and plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # LayeredNetworkGraph([g, h, i], node_labels=node_labels, ax=ax, layout=nx.spring_layout)
    graph_model = LayeredNetworkGraph([g, h, i], node_labels=node_labels,
                                      interaction_matrices=interaction_matrices,
                                      ax=ax, layout=nx.spring_layout)

    ani = matplotlib.animation.FuncAnimation(fig, graph_model.draw, frames=6, interval=1000, repeat=True)

    def onClick(event):
        if event.key == "p":
            global pause
            pause ^= True

            if pause:
                ani.event_source.stop()
            else:
                ani.event_source.start()

    fig.canvas.mpl_connect('key_press_event', onClick)

    plt.show()


if __name__ == '__main__':
    main()
