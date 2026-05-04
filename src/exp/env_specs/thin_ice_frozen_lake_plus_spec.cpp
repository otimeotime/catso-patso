#include "exp/experiment_spec.h"

#include "env/thin_ice_frozen_lake_plus_env.h"
#include "exp/env_specs/generic_discrete_env_spec_helpers.h"

#include <memory>
#include <ostream>

namespace mcts::exp {
    ExperimentSpec make_thin_ice_frozen_lake_plus_spec() {
        ExperimentSpec spec;

        spec.env_name = "thin_ice_frozen_lake_plus";
        spec.display_name = "ThinIceFrozenLakePlus";
        spec.output_prefix = "results_thin_ice_frozen_lake_plus";
        spec.tune_output_csv = "tune_thin_ice_frozen_lake_plus.csv";
        spec.horizon = 640;
        spec.eval_rollouts = 200;
        spec.runs = 3;
        spec.threads = 4;
        spec.base_seed = 4242;
        spec.cvar_tau = 0.05;
        spec.discount_gamma = 1.0;
        spec.trial_counts = {5000, 10000, 20000};

        mcts::exp::env_specs::configure_uct_family_run_config(
            spec.run_candidate_config,
            0.1,
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            true,
            1.0);
        mcts::exp::env_specs::configure_distributional_run_config(
            spec.run_candidate_config,
            51,
            1.0,
            64,
            1.0,
            1.0);

        mcts::exp::env_specs::configure_uct_family_tuning_grid(
            spec.tuning_grid_config,
            {mcts::UctManagerArgs::USE_AUTO_BIAS, 2.0, 5.0, 10.0},
            {0.05, 0.1, 0.2, 0.5},
            {1.0, 2.0, 4.0, 8.0});
        mcts::exp::env_specs::configure_distributional_tuning_grid(
            spec.tuning_grid_config,
            {25, 51, 100},
            {2.0, 4.0, 8.0},
            {32, 64, 128},
            {2.0, 4.0, 8.0},
            {1.0, 2.0, 4.0, 8.0},
            {0.05, 0.1, 0.2, 0.25});

        mcts::exp::env_specs::apply_generic_tuning_defaults(spec);

        spec.make_env = []() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::ThinIceFrozenLakePlusEnv>();
        };

        spec.is_catastrophic_transition = [](
            std::shared_ptr<const mcts::State>,
            std::shared_ptr<const mcts::Action>,
            std::shared_ptr<const mcts::State> next_state)
        {
            const auto typed_state = std::static_pointer_cast<const mcts::Int4TupleState>(next_state);
            return std::get<3>(typed_state->state) == mcts::exp::ThinIceFrozenLakePlusEnv::broken_status;
        };

        spec.evaluate_root_metrics =
            mcts::exp::env_specs::make_generic_root_metrics_evaluator<
                mcts::exp::ThinIceFrozenLakePlusEnv,
                mcts::Int4TupleState>();

        spec.print_env_info = [](std::ostream& os) {
            os << "[exp] ThinIceFrozenLakePlus"
               << ", N=8"
               << ", max_steps=640"
               << ", slip_prob=0.15"
               << ", break_prob=0.1"
               << ", cvar_tau=0.05"
               << "\n";
        };

        return spec;
    }
}
