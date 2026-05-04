#include "exp/experiment_spec.h"

#include "env/gambler_jackpot_env.h"
#include "exp/env_specs/generic_discrete_env_spec_helpers.h"

#include <memory>
#include <ostream>

namespace mcts::exp {
    ExperimentSpec make_gambler_jackpot_spec() {
        ExperimentSpec spec;

        spec.env_name = "gambler_jackpot";
        spec.display_name = "GamblerJackpot";
        spec.output_prefix = "results_gambler_jackpot";
        spec.tune_output_csv = "tune_gambler_jackpot.csv";
        spec.horizon = 1000;
        spec.eval_rollouts = 200;
        spec.runs = 3;
        spec.threads = 8;
        spec.base_seed = 4242;
        spec.cvar_tau = 0.1;
        spec.discount_gamma = 1.0;
        spec.trial_counts.reserve(20);
        for (int i = 1; i <= 20; ++i) {
            spec.trial_counts.push_back(i * 1000);
        }

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
            return std::make_shared<mcts::exp::GamblerJackpotEnv>();
        };

        spec.is_catastrophic_transition = [](
            std::shared_ptr<const mcts::State>,
            std::shared_ptr<const mcts::Action>,
            std::shared_ptr<const mcts::State> next_state)
        {
            const auto typed_state = std::static_pointer_cast<const mcts::Int3TupleState>(next_state);
            return std::get<2>(typed_state->state) == mcts::exp::GamblerJackpotEnv::bankrupt_status;
        };

        spec.evaluate_root_metrics =
            mcts::exp::env_specs::make_generic_root_metrics_evaluator<
                mcts::exp::GamblerJackpotEnv,
                mcts::Int3TupleState>();

        spec.print_env_info = [](std::ostream& os) {
            os << "[exp] GamblerJackpot"
               << ", capital_target=50"
               << ", max_steps=1000"
               << ", jackpot_prob=0.03"
               << ", cvar_tau=0.1"
               << "\n";
        };

        return spec;
    }
}
