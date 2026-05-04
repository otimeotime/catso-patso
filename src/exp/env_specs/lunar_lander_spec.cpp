#include "exp/experiment_spec.h"

#include "env/lunar_lander_env.h"
#include "exp/env_specs/generic_discrete_env_spec_helpers.h"

#include <memory>
#include <ostream>

namespace mcts::exp {
    namespace {
        struct LunarLanderEnvConfig {
            int max_steps = 200;
            double reward_normalisation = 256.0;
        };
    }

    ExperimentSpec make_lunar_lander_spec() {
        ExperimentSpec spec;
        const LunarLanderEnvConfig env_config{};

        spec.env_name = "lunar_lander";
        spec.display_name = "LunarLander";
        spec.output_prefix = "results_lunar_lander";
        spec.tune_output_csv = "tune_lunar_lander.csv";
        spec.horizon = env_config.max_steps;
        spec.eval_rollouts = 100;
        spec.runs = 3;
        spec.threads = 8;
        spec.base_seed = 4242;
        spec.cvar_tau = 0.05;
        spec.discount_gamma = 1.0;
        spec.trial_counts = {1000, 2000, 5000, 10000};

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

        spec.make_env = [env_config]() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::LunarLanderEnv>(
                env_config.max_steps,
                env_config.reward_normalisation);
        };

        spec.is_catastrophic_transition = [](
            std::shared_ptr<const mcts::State>,
            std::shared_ptr<const mcts::Action>,
            std::shared_ptr<const mcts::State> next_state)
        {
            const auto typed_state = std::static_pointer_cast<const mcts::exp::LunarLanderState>(next_state);
            return typed_state->status == mcts::exp::LunarLanderEnv::crashed_status;
        };

        spec.evaluate_root_metrics =
            mcts::exp::env_specs::make_generic_root_metrics_evaluator<
                mcts::exp::LunarLanderEnv,
                mcts::exp::LunarLanderState>();

        spec.print_env_info = [env_config](std::ostream& os) {
            os << "[exp] LunarLander"
               << ", max_steps=" << env_config.max_steps
               << ", reward_normalisation=" << env_config.reward_normalisation
               << ", cvar_tau=0.05"
               << "\n";
        };

        return spec;
    }
}
