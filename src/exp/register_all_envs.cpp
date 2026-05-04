#include "exp/env_registry.h"

namespace mcts::exp {
    ExperimentSpec make_bettinggame_spec();
    ExperimentSpec make_autonomous_vehicle_spec();
    ExperimentSpec make_gambler_jackpot_spec();
    ExperimentSpec make_guarded_maze_spec();
    ExperimentSpec make_laser_tag_safe_grid_spec();
    ExperimentSpec make_lunar_lander_spec();
    ExperimentSpec make_overflow_queue_spec();
    ExperimentSpec make_risky_ladder_spec();
    ExperimentSpec make_risky_shortcut_gridworld_spec();
    ExperimentSpec make_river_swim_stochastic_spec();
    ExperimentSpec make_thin_ice_frozen_lake_plus_spec();
    ExperimentSpec make_two_level_risky_treasure_spec();
    ExperimentSpec make_two_path_ssp_deceptive_spec();

    void register_all_envs() {
        EnvRegistry::instance().register_env("autonomous_vehicle", make_autonomous_vehicle_spec);
        EnvRegistry::instance().register_env("bettinggame", make_bettinggame_spec);
        EnvRegistry::instance().register_env("gambler_jackpot", make_gambler_jackpot_spec);
        EnvRegistry::instance().register_env("guarded_maze", make_guarded_maze_spec);
        EnvRegistry::instance().register_env("laser_tag_safe_grid", make_laser_tag_safe_grid_spec);
        EnvRegistry::instance().register_env("lunar_lander", make_lunar_lander_spec);
        EnvRegistry::instance().register_env("overflow_queue", make_overflow_queue_spec);
        EnvRegistry::instance().register_env("risky_ladder", make_risky_ladder_spec);
        EnvRegistry::instance().register_env("risky_shortcut_gridworld", make_risky_shortcut_gridworld_spec);
        EnvRegistry::instance().register_env("river_swim_stochastic", make_river_swim_stochastic_spec);
        EnvRegistry::instance().register_env("thin_ice_frozen_lake_plus", make_thin_ice_frozen_lake_plus_spec);
        EnvRegistry::instance().register_env("two_level_risky_treasure", make_two_level_risky_treasure_spec);
        EnvRegistry::instance().register_env("two_path_ssp_deceptive", make_two_path_ssp_deceptive_spec);
    }
}
