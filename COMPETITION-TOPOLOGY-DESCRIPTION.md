# Topology Description (ECML 2026: Real-World Baselines Challenge)

The competition runs on a single topology (120 x 150) with 28 stations:

![competition_map.png](docs/competition_map.png)

All 28 stations are provided in [*stations.pkl*](reinforcement_learning/sampling/stations.pkl). 6 lines that are used in the evaliation are also given as examples in [*level_o_scenario_1.pkl*](reinforcement_learning/sampling/level_0_scenario_1.pkl).

The map is subdivided into five *scenes* (i.e. regions):

![competition_scenes.png](docs/competition_scenes.png)

The scenes correspond to the scenarios in each level with the exception of level 6, where all five scenarios use scene 5 (cf. [level configurations](https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html)).