# Starterkit for ECML 2026: Real-World Baselines Challenge

<img width="1842" height="1521" alt="image" src="https://github.com/user-attachments/assets/2fdbc48c-1820-43a5-9483-51e2448418fb" />


This repo is a starterkit for participating in the Real-World Baselines Challenge for the [2026 ECML](https://ecmlpkdd.org/2026/)
conference: [competition.flatland.cloud](https://competition.flatland.cloud).

The competition documentation is included in [Flatland Book](https://flatland-association.github.io/flatland-book/challenges/ecml2026.html) (see also
the [topology description](COMPETITION-TOPOLOGY-DESCRIPTION.md)).

## TL;DR; aka. First Submission

1. Fork (or clone to keep your solution private until the end of the competition) this repo and code.
   See [existing forks](https://github.com/flatland-association/ecml2026-starterkit/forks) for illustration.
2. Manually trigger gh action `docker`  under `https://github.com/<user/orga>/<forked repo name>/actions/`
3. Copy the docker image URL from `https://github.com/<user/orga>/<forked repo name>/pkgs/container/<forked repo name>` and give the *Flatland Competition*
   account access to the package in the repo's *Package settings* for private repos.
4. Go to https://competition.flatland.cloud and enter the docker image URL when creating a submission.

![Workflow.drawio.png](docs/Workflow.drawio.png)

See [STEP-BY-STEP_GUIDE](STEP-BY-STEP_GUIDE.md) contributed by <a href="https://github.com/aiAdrian" target="_blank">aiAdrian</a> :partying_face:

## Customizing Your Submission

* Customize policy `submission.my_policy.MyPolicy`:

```python
from typing import Any

from flatland.envs.RailEnvPolicy import RailEnvPolicy
from flatland.envs.rail_env_action import RailEnvActions
from flatland.utils.seeding import np_random


class MyPolicy(RailEnvPolicy):
    def __init__(self):
        super().__init__()
        self.np_random, _ = np_random(seed=42)

    # implement this method, called for each agent in sequence
    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return self.np_random.choice(5)

    # in addition, implement if you need to do some work before `act` is called on individual agents
    # def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
    #     ...
    #     return super().act_many(handles, observations)
```

* Customize observation builder `submission.my_observation_builder.MyObservationBuilder`:

```python
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder


class MyObservationBuilder(ObservationBuilder[Environment, bool]):
    def reset(self):
        pass

    def get(self, handle: int = 0) -> bool:
        return True
```

* Add `pip` dependencies to `submission/requirements.txt`.
* All resources under `submission/` are added to the Docker image (add checkpoints here and load from your policy).

### Train your model using reinforcement learning

If you use reinforcement learning to train your policy, update the files in the submission folder to load your model either by setting *MyPolicy* using
*policy_from_checkpoint.py* or a custom implementation.

There is an example for a trained checkpoint with a custom observation and policy. To use the pretrained checkpoint, add the following files from the
*/reinforcement-learning* folder to the */submission* folder:

* my_observation_builder.py
* my_policy.py
* checkpoint.pt
* requirements.txt

## Local Testing

See [checks.yaml](.github/workflows/checks.yaml) for full details.

### Single episode

```bash
docker build  -t submission/mysolution -f Dockerfile .
docker run submission/mysolution flatland-trajectory-generate-from-policy  --data-dir /tmp --callbacks-pkg flatland.callbacks.generate_movie_callbacks --callbacks-cls GenerateMovieCallbacks --rewards flatland.envs.rewards.ECML2026Rewards --env-path reinforcement-learning/sampling/level_0_scenario_1.pkl
```

Output:

```log
+ PYTHONPATH=/home/conda
+ flatland-trajectory-generate-from-policy --data-dir /tmp --callbacks-pkg flatland.callbacks.generate_movie_callbacks --callbacks-cls GenerateMovieCallbacks --rewards flatland.envs.rewards.ECML2026Rewards --env-path reinforcement-learning/sampling/level_0_scenario_1.pkl
100%|█████████▉| 493/494 [00:24<00:00, 19.77it/s]
Generating Thumbnail...
Generating Normal Video...
Videos :  /tmp/outputs/out.mp4 /tmp/outputs/out_thumb.mp4
```

### Curriculum evaluation

```bash
# empty and re-create local folder
rm -fR outputs
mkdir -p outputs

unzip -o reinforcement_learning/curriculum/example_curriculum.zip -d reinforcement_learning/curriculum/curriculum

# run docker with volume mapping
ENVS=(
   '00_scene_1_ll-2_a-1'
   '01_scene_4_ll-2_a-1'
   '02_scene_5_ll-2_a-1'
   '03_scene_1_ll-2_a-10'
   '04_scene_4_ll-2_a-10'
   '05_scene_5_ll-2_a-10'
   '06_scene_1_ll-2_a-25'
   '07_scene_4_ll-2_a-25'
   '08_scene_5_ll-2_a-25'
   '09_scene_1_ll-3_a-1'
   '10_scene_4_ll-3_a-1'
   '11_scene_5_ll-3_a-1'
   '12_scene_1_ll-3_a-10'
   '13_scene_4_ll-3_a-10'
   '14_scene_5_ll-3_a-10'
   '15_scene_1_ll-3_a-25'
   '16_scene_4_ll-3_a-25'
   '17_scene_5_ll-3_a-25'
   '18_scene_1_ll-4_a-1'
   '19_scene_4_ll-4_a-1'
   '20_scene_5_ll-4_a-1'
   '21_scene_1_ll-4_a-10'
   '22_scene_4_ll-4_a-10'
   '23_scene_5_ll-4_a-10'
   '24_scene_1_ll-4_a-25'
   '25_scene_4_ll-4_a-25'
   '26_scene_5_ll-4_a-25'
)

for env in "${ENVS[@]}"; do
  echo "env: $env"
  mkdir -p ./outputs/${env}
  docker run -v ./reinforcement_learning/curriculum/curriculum:/inputs -v ./outputs:/tmp submission/mysolution flatland-trajectory-generate-from-policy --data-dir /tmp/${env} --rewards flatland.envs.rewards.ECML2026Rewards --env-path /inputs/${env}.pkl --ep-id ${env}
done
```

### Get report

```shell
# empty and re-create local folder
rm -fR analysis
mkdir -p analysis

# run docker with volume mapping
docker run -v ./outputs:/outputs -v ./analysis:/analysis submission/mysolution flatland-trajectory-analysis --root-data-dir /outputs --output-dir /analysis
# ls -al analysis
cat analysis/all_trains_arrived.csv
```

### Further CLI options

See the options for number of agents, grid size etc.:

```bash
docker run submission/mysolution flatland-trajectory-generate-from-policy --help
docker run submission/mysolution flatland-trajectory-generate-from-metadata --help
```

### Local environment

If you want to run the above commands in a local environment directly (independent of Docker container),
use [environment.yml from flatland-baselines](https://github.com/flatland-association/flatland-baselines/blob/main/environment.yml)
(see [instructions](https://github.com/flatland-association/flatland-baselines/tree/main?tab=readme-ov-file#tldr)).

## Further Information

[Flatland Benchmarks](https://github.com/flatland-association/flatland-benchmarks) (FAB) is an open-source web-based platform for running Benchmarks to foster
Open Research.

See [FAB User Guide](https://github.com/flatland-association/flatland-benchmarks/blob/main/docs/USER_GUIDE.md).
