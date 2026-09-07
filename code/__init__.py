"""
tab Game Engine - Python implementation of the ancient board game.

Modules:
    tab_game: Core game state, board, rules, dice simulation
    tab_ai:   AI agents (GA, Random) with original and expert fitness
    main:     Game runner, tournament system

Authors: Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh - Mutah University, Jordan
"""

from .tab_game import (
    Board, TabRules, StickDice, TurnAllocation, SoldierMove,
    PLAYER_1, PLAYER_2, STICK_VALUES, THROW_AGAIN
)
from .tab_ai import (
    GAAgent, RandomAgent, TabAgent, GAEngine,
    FitnessEvaluator, ExpertFitnessEvaluator
)

__version__ = "2.0.0"
__author__ = "Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh"
