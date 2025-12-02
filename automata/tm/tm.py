"""Classes and methods for working with all Turing machines."""

import abc
import os
from collections import defaultdict
from typing import AbstractSet, Literal, Tuple, Union

import automata.base.exceptions as exceptions
from automata.base.automaton import Automaton, AutomatonStateT
from automata.base.utils import (
    LayoutMethod,
    _missing_visual_imports,
    create_graph,
    create_unique_random_id,
    save_graph,
)

# Optional imports for use with visual functionality
if not _missing_visual_imports:
    import pygraphviz as pgv

TMStateT = AutomatonStateT
TMDirectionT = Literal["L", "R", "N"]


class TM(Automaton, metaclass=abc.ABCMeta):
    """An abstract base class for Turing machines."""

    __slots__ = tuple()

    tape_symbols: AbstractSet[str]
    blank_symbol: str

    @staticmethod
    def _get_edge_name(
        input_symbol: str = "", write_symbol: str = "", move_direction: str = ""
    ) -> str:
        return f"{input_symbol}/{write_symbol}/{move_direction}"


    def _read_input_symbol_subset(self) -> None:
        if not (self.input_symbols < self.tape_symbols):
            raise exceptions.MissingSymbolError(
                "The set of tape symbols is missing symbols from the input "
                "symbol set ({})".format(self.tape_symbols - self.input_symbols)
            )

    def _validate_blank_symbol(self) -> None:
        """Raise an error if blank symbol is not a tape symbol."""
        if self.blank_symbol not in self.tape_symbols:
            raise exceptions.InvalidSymbolError(
                "blank symbol {} is not a tape symbol".format(self.blank_symbol)
            )

    def _validate_nonfinal_initial_state(self) -> None:
        """Raise an error if the initial state is a final state."""
        if self.initial_state in self.final_states:
            raise exceptions.InitialStateError(
                "initial state {} cannot be a final state".format(self.initial_state)
            )


    def show_diagram(
        self,
        path: Union[str, os.PathLike, None] = None,
        *,
        layout_method: LayoutMethod = "dot",
        horizontal: bool = True,
        reverse_orientation: bool = False,
        fig_size: Union[Tuple[float, float], Tuple[float], None] = None,
        font_size: float = 14.0,
        arrow_size: float = 0.85,
        state_separation: float = 0.5,
    ) -> pgv.AGraph:
        """
        Generates a diagram of the associated TM.

        Parameters
        ----------
        path : Union[str, os.PathLike, None], default: None
            Path to output file. If None, the output will not be saved.
        horizontal : bool, default: True
            Direction of node layout in the output graph.
        reverse_orientation : bool, default: False
            Reverse direction of node layout in the output graph.
        fig_size : Union[Tuple[float, float], Tuple[float], None], default: None
            Figure size.
        font_size : float, default: 14.0
            Font size in the output graph.
        arrow_size : float, default: 0.85
            Arrow size in the output graph.
        state_separation : float, default: 0.5
            Distance between nodes in the output graph.

        Returns
        ------
        AGraph
            A diagram of the given automaton.
        """

        if _missing_visual_imports:
            raise _missing_visual_imports

        # Defining the graph.
        graph = create_graph(
            horizontal, reverse_orientation, fig_size, state_separation
        )

        font_size_str = str(font_size)
        arrow_size_str = str(arrow_size)

        # create unique id to avoid colliding with other states
        null_node = create_unique_random_id()

        graph.add_node(
            null_node,
            label="",
            tooltip=".",
            shape="point",
            fontsize=font_size_str,
        )
        initial_node = self._get_state_name(self.initial_state)
        graph.add_edge(
            null_node,
            initial_node,
            tooltip="->" + initial_node,
            arrowsize=arrow_size_str,
        )

        nonfinal_states = map(self._get_state_name, self.states - self.final_states)
        final_states = map(self._get_state_name, self.final_states)
        graph.add_nodes_from(nonfinal_states, shape="circle", fontsize=font_size_str)
        graph.add_nodes_from(final_states, shape="doublecircle", fontsize=font_size_str)

        is_edge_drawn = defaultdict(lambda: False)
        edge_labels = defaultdict(list)
        for (from_state, to_state, input_symbol,
             write_symbol, move_direction) in self.iter_transitions():
            if is_edge_drawn[from_state, to_state, input_symbol,
                             write_symbol, move_direction]:
                continue

            from_node = self._get_state_name(from_state)
            to_node = self._get_state_name(to_state)
            label = self._get_edge_name(input_symbol, write_symbol, move_direction)
            edge_labels[from_node, to_node].append(label)

        for (from_node, to_node), labels in edge_labels.items():
            graph.add_edge(
                from_node,
                to_node,
                label=",".join(sorted(labels)),
                arrowsize=arrow_size_str,
                fontsize=font_size_str,
            )

        # Set layout
        graph.layout(prog=layout_method)

        # Write diagram to file
        if path is not None:
            save_graph(graph, path)

        return graph
