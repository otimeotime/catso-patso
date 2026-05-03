#include "exp/experiment_spec.h"

#include "env/betting_game_env.h"
#include "exp/oracles/bettinggame_cvar_oracle.h"

#include <map>
#include <memory>
#include <ostream>

namespace mcts::exp {
    ExperimentSpec make_bettinggame_spec() {
        ExperimentSpec spec;

        constexpr double kWinProb = 0.8;
        constexpr int kMaxSequenceLength = 6;
        constexpr double kCvarTau = 0.20;
        constexpr int kEvalRollouts = 50;
        constexpr int kRuns = 3;
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
        spec.trial_counts = {2000, 5000, 10000, 15000};

        spec.run_candidate_config.catso_n_atoms = 100;
        spec.run_candidate_config.catso_optimism = 4.0;
        spec.run_candidate_config.power_mean_exponent = 1.0;
        spec.run_candidate_config.patso_particles = 128;
        spec.run_candidate_config.patso_optimism = 2.0;
        spec.run_candidate_config.uct_epsilon = 0.05;

        spec.tuning_grid_config.catso_n_atoms_values = {25, 51, 100};
        spec.tuning_grid_config.catso_optimism_values = {2.0, 4.0, 8.0};
        spec.tuning_grid_config.patso_particles_values = {32, 64, 128};
        spec.tuning_grid_config.patso_optimism_values = {2.0, 4.0, 8.0};
        spec.tuning_grid_config.tau_values = {0.05, 0.1, 0.25};
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
            std::make_shared<std::map<double, std::shared_ptr<mcts::exp::oracles::BettingGameCvarOracle>>>();
        spec.evaluate_root_metrics =
            [oracle_cache](
                std::shared_ptr<const mcts::MctsEnv> generic_env,
                std::shared_ptr<const mcts::MctsDNode> root,
                double eval_tau)
            {
                const auto env = std::static_pointer_cast<const mcts::exp::BettingGameEnv>(generic_env);
                auto& oracle = (*oracle_cache)[eval_tau];
                if (!oracle) {
                    oracle = std::make_shared<mcts::exp::oracles::BettingGameCvarOracle>(env, eval_tau);
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
