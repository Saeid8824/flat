import sys
from pathlib import Path

# 1. Automatically resolve paths so the Docker container can find the experimental folder
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
experimental_dir = root_dir / "experimental" / "flatland_solver"

if str(experimental_dir) not in sys.path:
    sys.path.insert(0, str(experimental_dir))

# 2. Import the actual MAPPO Policy you used during training
from policy.mappo.policy import MAPPOPolicy

# 3. Create the evaluation wrapper class the competition server expects
class MyPolicy(MAPPOPolicy):
    def __init__(self):
        # Dynamically point to the checkpoint.pt file inside this submission folder
        ckpt_path = current_dir / "checkpoint.pt"
        
        # Initialize the MAPPO model with the correct checkpoint
        # (This automatically sets up the correct architecture and 22-dim inputs)
        super().__init__(seed=42, checkpoint_path=str(ckpt_path))

    # The evaluator will automatically call the act() or act_many() 
    # functions that are inherited directly from MAPPOPolicy.