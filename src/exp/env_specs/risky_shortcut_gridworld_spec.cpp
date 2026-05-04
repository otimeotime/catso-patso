#include "exp/experiment_spec.h"

#include "env/risky_shortcut_gridworld_env.h"
#include "exp/env_specs/generic_discrete_env_spec_helpers.h"

#include <memory>
#include <ostream>
#include <vector>

namespace mcts::exp {
    ExperimentSpec make_risky_shortcut_gridworld_spec() {
        ExperimentSpec spec;

        constexpr int kGridSize = 4;
        constexpr double kSlipProb = 0.0;
        constexpr double kWindProb = 0.02;
        constexpr double kStepCost = 0.0;
        constexpr double kGoalReward = 5.0;
        constexpr double kCliffPenalty = -20.0;
        constexpr int kMaxSteps = 20;


        spec.env_name = "risky_shortcut_gridworld";
        spec.display_name = "RiskyShortcutGridworld";
        spec.output_prefix = "results_risky_shortcut_gridworld";
        spec.tune_output_csv = "tune_risky_shortcut_gridworld.csv";
        spec.horizon = kMaxSteps;
        spec.eval_rollouts = 200;
        spec.runs = 50;
        spec.threads = 8;
        spec.base_seed = 4242;
        spec.cvar_tau = 0.05;
        spec.discount_gamma = 0.95;
        spec.trial_counts = {100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};

        mcts::exp::env_specs::configure_uct_family_run_config(
            spec.run_candidate_config,
            0.1,
            mcts::UctManagerArgs::USE_AUTO_BIAS,
            true,
            1.0);
        mcts::exp::env_specs::configure_distributional_run_config(
            spec.run_candidate_config,
            100,
            2.0,
            100,
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
            {64, 100, 128},
            {2.0, 4.0, 8.0},
            {1.0},
            {0.05, 0.1, 0.2, 0.25});

        mcts::exp::env_specs::apply_generic_tuning_defaults(spec);

        spec.make_env = [=]() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::RiskyShortcutGridworldEnv>(
                kGridSize,
                kSlipProb,
                kWindProb,
                std::vector<int>{},
                kStepCost,
                kGoalReward,
                kCliffPenalty,
                kMaxSteps);
        };

        spec.is_catastrophic_transition = [=](
            std::shared_ptr<const mcts::State>,
            std::shared_ptr<const mcts::Action>,
            std::shared_ptr<const mcts::State> next_state)
        {
            const auto typed_state = std::static_pointer_cast<const mcts::Int3TupleState>(next_state);
            const int row = std::get<0>(typed_state->state);
            const int col = std::get<1>(typed_state->state);
            return row == kGridSize - 1 && col > 0 && col < kGridSize - 1;
        };

        spec.evaluate_root_metrics =
            mcts::exp::env_specs::make_generic_root_metrics_evaluator<
                mcts::exp::RiskyShortcutGridworldEnv,
                mcts::Int3TupleState>();

        spec.print_env_info = [=](std::ostream& os) {
            os << "[exp] RiskyShortcutGridworld"
               << ", grid_size=" << kGridSize
               << ", max_steps=" << kMaxSteps
               << ", slip_prob=" << kSlipProb
               << ", wind_prob=" << kWindProb
               << ", cvar_tau=" << spec.cvar_tau
               << "\n";
        };

        return spec;
    }
}
