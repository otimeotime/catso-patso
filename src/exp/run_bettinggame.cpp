#include "env/betting_game_env.h"

#include "algorithms/catso/catso_chance_node.h"
#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/uct/uct_chance_node.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env_context.h"
#include "exp/manager_config_printer.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace {
    constexpr double kCvarTolerance = 1e-12;
    constexpr double kCatastrophicBankrollThreshold = 1e-12;

    struct Candidate {
        string algo;
        string config;
        function<pair<shared_ptr<mcts::MctsManager>, shared_ptr<mcts::MctsDNode>>(
            shared_ptr<mcts::MctsEnv> env,
            shared_ptr<const mcts::State> init_state,
            int max_depth,
            int seed)> make;
    };

    struct EvalStats {
        double mean;
        double stddev;
        int catastrophic_count;
    };

    using ReturnDistribution = map<double, double>;

    struct BettingGameStateKey {
        double bankroll;
        int time;

        bool operator==(const BettingGameStateKey& other) const {
            return bankroll == other.bankroll && time == other.time;
        }
    };

    struct BettingGameStateKeyHash {
        size_t operator()(const BettingGameStateKey& key) const {
            size_t seed = std::hash<double>{}(key.bankroll);
            seed ^= std::hash<int>{}(key.time) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    struct OptimalDistributionSolution {
        ReturnDistribution state_value_distribution;
        unordered_map<int, ReturnDistribution> action_value_distributions;
        unordered_map<int, double> action_cvars;
        vector<int> optimal_actions;
        int canonical_action = -1;
        double optimal_cvar = 0.0;
    };

    struct BettingGameMetrics {
        double cvar_regret;
        double optimal_action_hit;
    };

    struct SummaryAccumulator {
        int count = 0;
        int regret_count = 0;
        double sum_mc_mean = 0.0;
        double sum_mc_stddev = 0.0;
        double sum_cvar_regret = 0.0;
        double sum_optimal_action_hit = 0.0;
        double sum_catastrophic_count = 0.0;
    };

    double compute_lower_tail_cvar(const ReturnDistribution& distribution, double tau) {
        if (distribution.empty()) {
            return 0.0;
        }

        const double tau_clamped = max(kCvarTolerance, min(tau, 1.0));
        double total_mass = 0.0;
        for (const auto& [value, probability] : distribution) {
            (void)value;
            if (probability > 0.0) {
                total_mass += probability;
            }
        }

        if (total_mass <= 0.0) {
            return 0.0;
        }

        const double tail_mass = tau_clamped * total_mass;
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;

        for (const auto& [value, probability] : distribution) {
            if (probability <= 0.0) {
                continue;
            }

            const double remaining_mass = max(0.0, tail_mass - cumulative_mass);
            const double mass_to_take = min(probability, remaining_mass);
            tail_sum += mass_to_take * value;
            cumulative_mass += probability;

            if (cumulative_mass >= tail_mass) {
                break;
            }
        }

        return tail_sum / tail_mass;
    }

    class BettingGameCvarOracle {
        private:
            shared_ptr<const mcts::exp::BettingGameEnv> env;
            double tau;
            unordered_map<BettingGameStateKey, OptimalDistributionSolution, BettingGameStateKeyHash> memo;

            static BettingGameStateKey make_key(shared_ptr<const mcts::exp::BettingGameState> state) {
                return {state->bankroll, state->time};
            }

            static void add_weighted_shifted_distribution(
                ReturnDistribution& out,
                const ReturnDistribution& in,
                double weight,
                double shift)
            {
                for (const auto& [value, probability] : in) {
                    out[value + shift] += weight * probability;
                }
            }

        public:
            BettingGameCvarOracle(shared_ptr<const mcts::exp::BettingGameEnv> env, double tau) :
                env(move(env)),
                tau(tau)
            {
            }

            const OptimalDistributionSolution& solve_state(shared_ptr<const mcts::exp::BettingGameState> state) {
                const BettingGameStateKey key = make_key(state);
                const auto memo_it = memo.find(key);
                if (memo_it != memo.end()) {
                    return memo_it->second;
                }

                OptimalDistributionSolution solution;
                if (env->is_sink_state(state)) {
                    solution.state_value_distribution[0.0] = 1.0;
                    auto [it, inserted] = memo.emplace(key, move(solution));
                    (void)inserted;
                    return it->second;
                }

                auto actions = env->get_valid_actions(state);
                for (const auto& action : *actions) {
                    ReturnDistribution action_distribution;
                    auto transitions = env->get_transition_distribution(state, action);
                    for (const auto& [next_state, probability] : *transitions) {
                        const auto& child_solution = solve_state(next_state);
                        const double local_reward = env->get_reward(state, action, next_state);
                        add_weighted_shifted_distribution(
                            action_distribution,
                            child_solution.state_value_distribution,
                            probability,
                            local_reward);
                    }

                    const int action_id = action->action;
                    const double action_cvar = compute_lower_tail_cvar(action_distribution, tau);
                    solution.action_value_distributions.emplace(action_id, action_distribution);
                    solution.action_cvars.emplace(action_id, action_cvar);

                    if (solution.optimal_actions.empty() || action_cvar > solution.optimal_cvar + kCvarTolerance) {
                        solution.optimal_cvar = action_cvar;
                        solution.optimal_actions = {action_id};
                        solution.canonical_action = action_id;
                    }
                    else if (abs(action_cvar - solution.optimal_cvar) <= kCvarTolerance) {
                        solution.optimal_actions.push_back(action_id);
                        solution.canonical_action = min(solution.canonical_action, action_id);
                    }
                }

                if (solution.canonical_action >= 0) {
                    solution.state_value_distribution = solution.action_value_distributions.at(solution.canonical_action);
                }

                auto [it, inserted] = memo.emplace(key, move(solution));
                (void)inserted;
                return it->second;
            }
    };

    static EvalStats evaluate_tree(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        int horizon,
        int rollouts,
        int threads,
        int seed)
    {
        (void)threads;

        mcts::RandManager eval_rng(seed);
        mcts::EvalPolicy policy(root, env, eval_rng);
        vector<double> sampled_returns;
        sampled_returns.reserve(static_cast<size_t>(max(rollouts, 0)));
        int catastrophic_count = 0;

        for (int rollout = 0; rollout < rollouts; ++rollout) {
            policy.reset();

            int num_actions_taken = 0;
            double sample_return = 0.0;
            shared_ptr<const mcts::State> state = env->get_initial_state_itfc();
            auto context_ptr = env->sample_context_itfc(state);
            mcts::MctsEnvContext& context = *context_ptr;

            while (num_actions_taken < horizon && !env->is_sink_state_itfc(state)) {
                shared_ptr<const mcts::Action> action = policy.get_action(state, context);
                shared_ptr<const mcts::State> next_state =
                    env->sample_transition_distribution_itfc(state, action, eval_rng);
                shared_ptr<const mcts::Observation> observation =
                    env->sample_observation_distribution_itfc(action, next_state, eval_rng);

                sample_return += env->get_reward_itfc(state, action, observation);
                policy.update_step(action, observation);
                state = next_state;
                num_actions_taken++;
            }

            sampled_returns.push_back(sample_return);

            shared_ptr<const mcts::exp::BettingGameState> final_state =
                static_pointer_cast<const mcts::exp::BettingGameState>(state);
            if (final_state->bankroll <= kCatastrophicBankrollThreshold) {
                catastrophic_count++;
            }
        }

        if (sampled_returns.empty()) {
            return {0.0, 0.0, 0};
        }

        double mean = 0.0;
        for (double sample_return : sampled_returns) {
            mean += sample_return;
        }
        mean /= static_cast<double>(sampled_returns.size());

        double stddev = 0.0;
        if (sampled_returns.size() > 1u) {
            double variance = 0.0;
            for (double sample_return : sampled_returns) {
                variance += pow(sample_return - mean, 2.0);
            }
            variance /= static_cast<double>(sampled_returns.size() - 1u);
            stddev = sqrt(variance);
        }

        return {mean, stddev, catastrophic_count};
    }

    static BettingGameMetrics evaluate_bettinggame_metrics(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution)
    {
        auto context = env->sample_context_itfc(env->get_initial_state_itfc());
        shared_ptr<const mcts::Action> recommended_action = root->recommend_action_itfc(*context);
        const int recommended_action_id =
            static_pointer_cast<const mcts::IntAction>(recommended_action)->action;

        BettingGameMetrics metrics{
            numeric_limits<double>::quiet_NaN(),
            0.0
        };

        for (int optimal_action_id : root_solution.optimal_actions) {
            if (recommended_action_id == optimal_action_id) {
                metrics.optimal_action_hit = 1.0;
                break;
            }
        }

        auto catso_root = dynamic_pointer_cast<const mcts::CatsoDNode>(root);
        if (catso_root != nullptr && catso_root->has_child_node(recommended_action)) {
            const double estimated_recommended_cvar =
                catso_root->get_child_node(recommended_action)->get_cvar_value();
            metrics.cvar_regret = root_solution.optimal_cvar - estimated_recommended_cvar;
            return metrics;
        }

        auto uct_root = dynamic_pointer_cast<const mcts::UctDNode>(root);
        if (uct_root != nullptr && uct_root->has_child_node(recommended_action)) {
            const double estimated_recommended_mean =
                uct_root->get_child_node(recommended_action)->get_mean_value();
            metrics.cvar_regret = root_solution.optimal_cvar - estimated_recommended_mean;
        }

        return metrics;
    }

    static vector<Candidate> build_candidates(double cvar_tau) {
        vector<Candidate> cands;

        cands.push_back({"UCT", "bias=auto,epsilon=0.1", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::UctManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.bias = mcts::UctManagerArgs::USE_AUTO_BIAS;
            args.epsilon_exploration = 0.1;
            args.seed = seed;
            auto mgr = make_shared<mcts::UctManager>(args);
            auto root = make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"CATSO", "n_atoms=51,optimism=1.0,p=2.0,tau=" + to_string(cvar_tau), [cvar_tau](auto env, auto init_state, int max_depth, int seed) {
            mcts::CatsoManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.n_atoms = 100;
            args.optimism_constant = 4.0;
            args.power_mean_exponent = 2.0;
            args.cvar_tau = cvar_tau;
            args.seed = seed;
            auto mgr = make_shared<mcts::CatsoManager>(args);
            auto root = make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"PATSO", "max_particles=64,optimism=1.0,p=2.0,tau=" + to_string(cvar_tau), [cvar_tau](auto env, auto init_state, int max_depth, int seed) {
            mcts::PatsoManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.max_particles = 100;
            args.optimism_constant = 4.0;
            args.power_mean_exponent = 2.0;
            args.cvar_tau = cvar_tau;
            args.seed = seed;
            auto mgr = make_shared<mcts::PatsoManager>(args);
            auto root = make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        return cands;
    }
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    const double win_prob = 0.8;
    const int max_sequence_length = 6;
    const double max_state_value = mcts::exp::BettingGameEnv::default_max_state_value;
    const double initial_state = mcts::exp::BettingGameEnv::default_initial_state;
    const double reward_normalisation = mcts::exp::BettingGameEnv::default_reward_normalisation;
    const double cvar_tau = 0.2;
    const int horizon = max_sequence_length;
    const int eval_rollouts = 200;
    const int runs = 5;
    const int threads = 8;
    const int base_seed = 4242;

    vector<int> trial_counts;
    for (int i = 1; i <= 30; ++i) {
        trial_counts.push_back(i * 1000);
    }

    auto env = make_shared<mcts::exp::BettingGameEnv>(
        win_prob,
        max_sequence_length,
        max_state_value,
        initial_state,
        reward_normalisation);
    auto init_state = env->get_initial_state_itfc();
    auto candidates = build_candidates(cvar_tau);
    BettingGameCvarOracle oracle(env, cvar_tau);
    const auto& root_solution = oracle.solve_state(env->get_initial_state());
    map<pair<string,int>, SummaryAccumulator> summary_by_algo_and_trial;

    ofstream out("results_bettinggame.csv", ios::out | ios::trunc);
    out << "env,algorithm,run,trial,mc_mean,mc_stddev,cvar_regret,optimal_action_hit,catastrophic_count\n";

    cout << "[exp] BettingGame"
         << ", horizon=" << horizon
         << ", win_prob=" << win_prob
         << ", max_sequence_length=" << max_sequence_length
         << ", initial_state=" << initial_state
         << ", max_state_value=" << max_state_value
         << ", cvar_tau=" << cvar_tau
         << "\n";
    cout << "  Algorithms: " << candidates.size()
         << ", Runs: " << runs
         << ", Trial counts: " << trial_counts.size()
         << "\n";
    cout << "  Root optimal CVaR: " << root_solution.optimal_cvar << "\n";

    for (const auto& cand : candidates) {
        bool printed_config = false;

        for (int run = 0; run < runs; ++run) {
            int seed = base_seed + 1000 * run + static_cast<int>(hash<string>{}(cand.algo) % 997);

            auto [mgr, root] = cand.make(env, init_state, horizon, seed);
            if (!printed_config) {
                const string runtime_config =
                    mcts::exp::describe_manager_config(static_pointer_cast<const mcts::MctsManager>(mgr));
                cout << "\nRunning " << cand.algo;
                if (!runtime_config.empty()) {
                    cout << " [" << runtime_config << "]";
                }
                else if (!cand.config.empty()) {
                    cout << " [" << cand.config << "]";
                }
                cout << "...\n";
                printed_config = true;
            }
            auto pool = make_unique<mcts::MctsPool>(mgr, root, threads);

            int trials_so_far = 0;
            for (int target_trials : trial_counts) {
                const int trials_to_add = target_trials - trials_so_far;
                if (trials_to_add > 0) {
                    pool->run_trials(trials_to_add, numeric_limits<double>::max(), true);
                    trials_so_far = target_trials;
                }

                auto stats = evaluate_tree(env, root, horizon, eval_rollouts, threads, seed + 777);
                auto metrics = evaluate_bettinggame_metrics(env, root, root_solution);
                out << "bettinggame," << cand.algo << "," << run
                    << "," << target_trials << "," << stats.mean << "," << stats.stddev
                    << "," << metrics.cvar_regret << "," << metrics.optimal_action_hit
                    << "," << stats.catastrophic_count << "\n";

                SummaryAccumulator& summary = summary_by_algo_and_trial[{cand.algo, target_trials}];
                summary.count++;
                summary.sum_mc_mean += stats.mean;
                summary.sum_mc_stddev += stats.stddev;
                summary.sum_optimal_action_hit += metrics.optimal_action_hit;
                summary.sum_catastrophic_count += static_cast<double>(stats.catastrophic_count);
                if (!std::isnan(metrics.cvar_regret)) {
                    summary.regret_count++;
                    summary.sum_cvar_regret += metrics.cvar_regret;
                }
            }

            cout << "  " << cand.algo << " run " << (run + 1) << "/" << runs << " done\n";
        }
    }

    ofstream summary_out("results_bettinggame_summary.csv", ios::out | ios::trunc);
    summary_out << "env,algorithm,trial,mc_mean,mc_stddev,cvar_regret,optimal_action_prob,catastrophic_count\n";
    for (const auto& [algo_trial, summary] : summary_by_algo_and_trial) {
        const string& algo = algo_trial.first;
        const int trial = algo_trial.second;
        const double mean_cvar_regret = (summary.regret_count > 0)
            ? (summary.sum_cvar_regret / static_cast<double>(summary.regret_count))
            : numeric_limits<double>::quiet_NaN();

        summary_out << "bettinggame," << algo << "," << trial
            << "," << (summary.sum_mc_mean / static_cast<double>(summary.count))
            << "," << (summary.sum_mc_stddev / static_cast<double>(summary.count))
            << "," << mean_cvar_regret
            << "," << (summary.sum_optimal_action_hit / static_cast<double>(summary.count))
            << "," << (summary.sum_catastrophic_count / static_cast<double>(summary.count))
            << "\n";
    }

    cout << "\nResults written to results_bettinggame.csv and results_bettinggame_summary.csv\n";
    return 0;
}
