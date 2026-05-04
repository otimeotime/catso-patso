#include "exp/experiment_spec.h"

#include "env/two_level_risky_treasure_env.h"
#include "exp/env_specs/generic_discrete_env_spec_helpers.h"

#include <memory>
#include <ostream>

namespace mcts::exp {
    ExperimentSpec make_two_level_risky_treasure_spec() {
        ExperimentSpec spec;

        spec.env_name = "two_level_risky_treasure";
        spec.display_name = "TwoLevelRiskyTreasure";
        spec.output_prefix = "results_two_level_risky_treasure";
        spec.tune_output_csv = "tune_two_level_risky_treasure.csv";
        spec.horizon = 2;
        spec.eval_rollouts = 200;
        spec.runs = 50;
        spec.threads = 8;
        spec.base_seed = 4242;
        spec.cvar_tau = 0.25;
        spec.discount_gamma = 1.0;
        spec.trial_counts = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000};

        mcts::exp::env_specs::configure_uct_family_run_config(
            spec.run_candidate_config,
            0.5,
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            true,
            1.0);
        mcts::exp::env_specs::configure_distributional_run_config(
            spec.run_candidate_config,
            25,
            2.0,
            128,
            2.0,
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
            {1.0},
            {0.05, 0.1, 0.2, 0.25});

        mcts::exp::env_specs::apply_generic_tuning_defaults(spec);

        spec.make_env = []() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::TwoLevelRiskyTreasureEnv>();
        };

        spec.is_catastrophic_transition = [](
            std::shared_ptr<const mcts::State>,
            std::shared_ptr<const mcts::Action>,
            std::shared_ptr<const mcts::State> next_state)
        {
            const auto typed_state = std::static_pointer_cast<const mcts::Int4TupleState>(next_state);
            return std::get<3>(typed_state->state) == mcts::exp::TwoLevelRiskyTreasureEnv::fail_status;
        };

        spec.evaluate_root_metrics =
            mcts::exp::env_specs::make_generic_root_metrics_evaluator<
                mcts::exp::TwoLevelRiskyTreasureEnv,
                mcts::Int4TupleState>();

        spec.print_env_info = [](std::ostream& os) {
            os << "[exp] TwoLevelRiskyTreasure"
               << ", stages=2"
               << ", prior_success_prob=0.4"
               << ", hint_error=0.3"
               << ", cvar_tau=0.25"
               << "\n";
        };

        return spec;
    }
}
