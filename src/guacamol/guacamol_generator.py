from typing import Optional, List

from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from src.sac.guac_train import run


class Generator(GoalDirectedGenerator):
    def __init__(self):
        pass

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        return run(scoring_function, number_molecules)
