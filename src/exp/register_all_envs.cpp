#include "exp/env_registry.h"

namespace mcts::exp {
    ExperimentSpec make_bettinggame_spec();
    ExperimentSpec make_autonomous_vehicle_spec();
    ExperimentSpec make_guarded_maze_spec();

    void register_all_envs() {
        EnvRegistry::instance().register_env("bettinggame", make_bettinggame_spec);
        EnvRegistry::instance().register_env("autonomous_vehicle", make_autonomous_vehicle_spec);
        EnvRegistry::instance().register_env("guarded_maze", make_guarded_maze_spec);
    }
}
