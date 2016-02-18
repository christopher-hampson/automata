#!/usr/bin/env python3

import abc


class Automaton(metaclass=abc.ABCMeta):
    """an abstract base class for finite automata"""

    @abc.abstractmethod
    def __init__(self, states, symbols, transitions, initial_state,
                 final_states):
        """initializes a complete finite automaton"""
        pass

    @abc.abstractmethod
    def validate_input(self):
        """returns True if the given string is accepted by this automaton;
        raises the appropriate exception otherwise"""
        pass

    @abc.abstractmethod
    def validate_automaton(self):
        """returns True if this automaton is internally consistent;
        raises the appropriate exception otherwise"""
        pass

    @staticmethod
    def stringify_states(states):
        """stringifies the given set of states as a single state name"""
        return '{{{}}}'.format(''.join(sorted(list(states))))


class AutomatonError(Exception):
    """the base class for all automaton-related errors"""
    pass


class InvalidStateError(AutomatonError):
    """a state is not a valid state for this automaton"""
    pass


class InvalidSymbolError(AutomatonError):
    """a symbol is not a valid symbol for this automaton"""
    pass


class MissingStateError(AutomatonError):
    """a state is missing from the transition function"""
    pass


class MissingSymbolError(AutomatonError):
    """a symbol is missing from the transition function"""
    pass


class FinalStateError(AutomatonError):
    """the automaton stopped at a non-final state"""
    pass
