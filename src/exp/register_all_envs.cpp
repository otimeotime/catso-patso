#include "exp/env_registry.h"

namespace mcts::exp {
    ExperimentSpec make_autonomous_vehicle_spec();
    ExperimentSpec make_risky_shortcut_gridworld_spec();
    ExperimentSpec make_two_level_risky_treasure_spec();

    void register_all_envs() {
        EnvRegistry::instance().register_env("autonomous_vehicle", make_autonomous_vehicle_spec);
        EnvRegistry::instance().register_env("risky_shortcut_gridworld", make_risky_shortcut_gridworld_spec);
        EnvRegistry::instance().register_env("two_level_risky_treasure", make_two_level_risky_treasure_spec);
    }
}
