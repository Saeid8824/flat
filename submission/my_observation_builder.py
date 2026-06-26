import sys
from pathlib import Path

# 1. Automatically resolve paths so the Docker container can find the experimental folder
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
experimental_dir = root_dir / "experimental" / "flatland_solver"

if str(experimental_dir) not in sys.path:
    sys.path.insert(0, str(experimental_dir))

# 2. Import the correct Observation Builder you used to train
from observations.decision_point_observation import DecisionPointObservation

# 3. Create the final class that matches the training setup
class FinalObservationBuilder(DecisionPointObservation):
    def __init__(self, **kwargs):
        # You trained with --obs-variant decision_point which defaults to search_depth=4.
        # We hardcode it here so the evaluator initializes it correctly.
        super().__init__(debug=False, search_depth=4)

    def _rail_get_transitions(self, pos, direction):
        # Compatibility patch to ensure the Flatland environment API 
        # transitions work correctly on the evaluation server.
        p = self._pos_tuple(pos)
        return self.env.rail.get_transitions((p, int(direction)))

# 4. Expose it under the exact variable name the competition evaluator expects
MyObservationBuilder = FinalObservationBuilder