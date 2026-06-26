import sys
from pathlib import Path

# 1. Automatically add the experimental folder to the Python path 
# so Docker can locate and import the training observation builder.
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
experimental_dir = root_dir / "experimental" / "flatland_solver"

if str(experimental_dir) not in sys.path:
    sys.path.insert(0, str(experimental_dir))

# 2. Import the exact Observation Builder you used to train (decision_point)
from observations.decision_point_observation import DecisionPointObservation

# 3. Create the final class that matches the training setup and includes the Flatland compatibility patch
class FinalObservationBuilder(DecisionPointObservation):
    def __init__(self, **kwargs):
        # You trained with --obs-variant decision_point which defaults to search_depth=4.
        # We hardcode it here so the evaluator initializes it correctly.
        super().__init__(debug=False, search_depth=4)

    def _rail_get_transitions(self, pos, direction):
        # Compatibility patch matching the starter kit's factory.py 
        # to ensure the Flatland environment API transitions work correctly.
        p = self._pos_tuple(pos)
        return self.env.rail.get_transitions((p, int(direction)))

# 4. Expose it under the exact variable name the competition evaluator expects
MyObservationBuilder = FinalObservationBuilder