#include "exp/experiment_spec.h"

#include "env/betting_game_env.h"
#include "exp/env_specs/generic_discrete_env_spec_helpers.h"
#include "exp/oracles/bettinggame_cvar_oracle.h"

#include <map>
#include <memory>
#include <ostream>

namespace mcts::exp {
    ExperimentSpec make_bettinggame_spec() {
        ExperimentSpec spec;

        constexpr double kWinProb = 0.8;
        constexpr int kMaxSequenceLength = 6;
        constexpr double kCvarTau = 0.25;
        constexpr int kEvalRollouts = 50;
        constexpr int kRuns = 50;
        constexpr int kThreads = 8;
        constexpr int kTuneThreads = 16;
        constexpr int kBaseSeed = 4242;

        spec.env_name = "bettinggame";
        spec.display_name = "BettingGame";
        spec.output_prefix = "results_bettinggame";
        spec.tune_output_csv = "tune_bettinggame.csv";
        spec.horizon = kMaxSequenceLength;
        spec.eval_rollouts = kEvalRollouts;
        spec.runs = kRuns;
        spec.threads = kThreads;
        spec.base_seed = kBaseSeed;
        spec.cvar_tau = kCvarTau;
        spec.discount_gamma = 1.0;
        spec.trial_counts = {100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};

        mcts::exp::env_specs::configure_uct_family_run_config(
            spec.run_candidate_config,
            0.05,
            2,
            false,
            1.0);
        mcts::exp::env_specs::configure_distributional_run_config(
            spec.run_candidate_config,
            50,
            8.0,
            32,
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
            {0.05, 0.1, 0.25});
        spec.tune_total_trials = 10000;
        spec.tune_runs = 1;
        spec.tune_threads = kTuneThreads;
        spec.progress_batch_trials = 10000;

        spec.make_env = [=]() -> std::shared_ptr<mcts::MctsEnv> {
            return std::make_shared<mcts::exp::BettingGameEnv>(
                kWinProb,
                kMaxSequenceLength,
                mcts::exp::BettingGameEnv::default_max_state_value,
                mcts::exp::BettingGameEnv::default_initial_state,
                mcts::exp::BettingGameEnv::default_reward_normalisation);
        };

        spec.print_env_info = [](std::ostream& os) {
            os << "[exp] BettingGame"
               << ", horizon=" << kMaxSequenceLength
               << ", win_prob=" << kWinProb
               << ", max_sequence_length=" << kMaxSequenceLength
               << ", initial_state=" << mcts::exp::BettingGameEnv::default_initial_state
               << ", max_state_value=" << mcts::exp::BettingGameEnv::default_max_state_value
               << ", cvar_tau=" << kCvarTau
               << "\n";
        };

        auto oracle_cache =
            std::make_shared<std::map<
                std::pair<double, double>,
                std::shared_ptr<mcts::exp::oracles::BettingGameCvarOracle>>>();
        spec.evaluate_root_metrics =
            [oracle_cache](
                std::shared_ptr<const mcts::MctsEnv> generic_env,
                std::shared_ptr<const mcts::MctsDNode> root,
                double eval_tau,
                double discount_gamma)
            {
                const auto env = std::static_pointer_cast<const mcts::exp::BettingGameEnv>(generic_env);
                auto& oracle = (*oracle_cache)[{eval_tau, discount_gamma}];
                if (!oracle) {
                    oracle = std::make_shared<mcts::exp::oracles::BettingGameCvarOracle>(
                        env,
                        eval_tau,
                        discount_gamma);
                }
                const auto& root_solution = oracle->solve_state(env->get_initial_state());
                return mcts::exp::oracles::evaluate_bettinggame_root_metrics(
                    env,
                    root,
                    root_solution,
                    eval_tau);
            };

        return spec;
    }
}
