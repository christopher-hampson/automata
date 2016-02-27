#!/usr/bin/env python3
"""Classes and methods for working with nondeterministic finite automata."""

import copy
import automata.automaton as automaton
import automata.dfa


class NFA(automaton.Automaton):
    """Create a nondeterministic finite automaton."""

    def __init__(self, obj=None, *, states=None, symbols=None,
                 transitions=None, initial_state=None, final_states=None):
        """Initialize a complete NFA."""
        if isinstance(obj, automata.dfa.DFA):
            self._init_from_dfa(obj)
        elif isinstance(obj, NFA):
            self._init_from_nfa(obj)
        else:
            self.states = states.copy()
            self.symbols = symbols.copy()
            self.transitions = copy.deepcopy(transitions)
            self.initial_state = initial_state
            self.final_states = final_states.copy()
            self.validate_automaton()

    def _init_from_nfa(self, nfa):
        """Initialize this NFA as an exact copy of the given NFA."""
        self.__init__(
            states=nfa.states, symbols=nfa.symbols,
            transitions=nfa.transitions, initial_state=nfa.initial_state,
            final_states=nfa.final_states)

    def _init_from_dfa(self, dfa):
        """Initialize this NFA as one equivalent to the given DFA."""
        nfa_transitions = {}
        for start_state, paths in dfa.transitions.items():
            nfa_transitions[start_state] = {}
            for symbol, end_state in paths.items():
                nfa_transitions[start_state][symbol] = {end_state}

        self.__init__(
            states=dfa.states, symbols=dfa.symbols,
            transitions=nfa_transitions, initial_state=dfa.initial_state,
            final_states=dfa.final_states)

    def _validate_transition_symbols(self, start_state, paths):
        """Raise an error if the transition symbols are missing or invalid."""
        path_symbols = set(paths.keys())
        invalid_symbols = path_symbols - self.symbols.union({''})
        if invalid_symbols:
            raise automaton.InvalidSymbolError(
                'state {} has invalid transition symbols ({})'.format(
                    start_state, ', '.join(invalid_symbols)))

    def validate_automaton(self):
        """Return True if this NFA is internally consistent."""
        self._validate_transition_start_states()

        for start_state, paths in self.transitions.items():

            self._validate_transition_symbols(start_state, paths)

            path_states = set().union(*paths.values())
            self._validate_transition_end_states(path_states)

        self._validate_initial_state()
        self._validate_final_states()

        return True

    def _get_lambda_closure(self, start_state):
        """
        Return the lambda closure for the given state.

        The lambda closure of a state q is the set containing q, along with
        every state that can be reached from q by following only lambda
        transitions.
        """
        stack = []
        encountered_states = set()
        stack.append(start_state)

        while stack:

            state = stack.pop()
            if state not in encountered_states:
                encountered_states.add(state)

                if '' in self.transitions[state]:
                    stack.extend(self.transitions[state][''])

        return encountered_states

    def _get_next_current_states(self, current_states, symbol):
        """Return the next set of current states given the current set."""
        next_current_states = set()

        for current_state in current_states:

            symbol_end_states = self.transitions[current_state].get(symbol)
            if symbol_end_states:
                for end_state in symbol_end_states:
                    next_current_states.update(
                        self._get_lambda_closure(end_state))

        print(next_current_states)
        return next_current_states

    def validate_input(self, input_str):
        """
        Check if the given string is accepted by this NFA.

        Return a set of states the NFA stopped at if string is valid.
        """
        current_states = self._get_lambda_closure(self.initial_state)

        for symbol in input_str:

            self._validate_input_symbol(symbol)
            current_states = self._get_next_current_states(
                current_states, symbol)

        if not (current_states & self.final_states):
            raise automaton.FinalStateError(
                'the automaton stopped at all non-final states ({})'.format(
                    ', '.join(current_states)))

        return current_states
