#include "env/betting_game_env.h"

#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace {
    constexpr double kCvarTolerance = 1e-12;
    constexpr double kDefaultOptimalActionRegretThreshold = 0.0;

    struct Candidate {
        string algo;
        string config;
        double eval_tau;
        function<pair<shared_ptr<mcts::MctsManager>, shared_ptr<mcts::MctsDNode>>(
            shared_ptr<mcts::MctsEnv> env,
            shared_ptr<const mcts::State> init_state,
            int max_depth,
            int seed)> make;
    };

    struct EvalStats {
        double mean;
        double stddev;
        double cvar;
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

    struct Aggregate {
        int count = 0;
        double sum_mc_mean = 0.0;
        double sum_mc_stddev = 0.0;
        double sum_mc_cvar = 0.0;
        double sum_cvar_regret = 0.0;
        double sum_optimal_action_hit = 0.0;
        double sum_catastrophic_count = 0.0;
    };

    static string csv_escape(const string& value) {
        if (value.find_first_of(",\"\n\r") == string::npos) {
            return value;
        }

        string escaped;
        escaped.reserve(value.size() + 2);
        escaped.push_back('"');
        for (char ch : value) {
            if (ch == '"') {
                escaped += "\"\"";
            }
            else {
                escaped.push_back(ch);
            }
        }
        escaped.push_back('"');
        return escaped;
    }

    static void run_trials_with_progress(
        mcts::MctsPool& pool,
        int total_trials,
        int batch_trials,
        const string& label)
    {
        const int actual_batch_trials = max(1, min(batch_trials, total_trials));
        int completed_trials = 0;
        auto start_time = chrono::steady_clock::now();

        while (completed_trials < total_trials) {
            const int trials_to_add = min(actual_batch_trials, total_trials - completed_trials);
            pool.run_trials(trials_to_add, numeric_limits<double>::max(), true);
            completed_trials += trials_to_add;

            const double progress = 100.0 * static_cast<double>(completed_trials) / static_cast<double>(total_trials);
            const double elapsed_seconds =
                chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - start_time).count();

            ostringstream oss;
            oss << "    " << label
                << " trials " << completed_trials << "/" << total_trials
                << " (" << fixed << setprecision(1) << progress << "%, "
                << setprecision(2) << elapsed_seconds << "s)";
            cout << oss.str() << endl;
        }
    }

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

    double compute_empirical_lower_tail_cvar(vector<double> samples, double tau) {
        if (samples.empty()) {
            return 0.0;
        }

        sort(samples.begin(), samples.end());

        const double tau_clamped = max(kCvarTolerance, min(tau, 1.0));
        const double tail_mass = tau_clamped * static_cast<double>(samples.size());
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;

        for (double sample : samples) {
            const double remaining_mass = max(0.0, tail_mass - cumulative_mass);
            const double mass_to_take = min(1.0, remaining_mass);
            tail_sum += mass_to_take * sample;
            cumulative_mass += 1.0;

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
        int seed,
        double cvar_tau)
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
            bool rollout_was_catastrophic = env->is_catastrophic_state_itfc(state);
            auto context_ptr = env->sample_context_itfc(state);
            mcts::MctsEnvContext& context = *context_ptr;

            while (num_actions_taken < horizon && !env->is_sink_state_itfc(state)) {
                shared_ptr<const mcts::Action> action = policy.get_action(state, context);
                shared_ptr<const mcts::State> next_state =
                    env->sample_transition_distribution_itfc(state, action, eval_rng);
                shared_ptr<const mcts::Observation> observation =
                    env->sample_observation_distribution_itfc(action, next_state, eval_rng);

                sample_return += env->get_reward_itfc(state, action, observation);
                if (env->is_catastrophic_state_itfc(next_state)) {
                    rollout_was_catastrophic = true;
                }
                policy.update_step(action, observation);
                state = next_state;
                num_actions_taken++;
            }

            sampled_returns.push_back(sample_return);
            if (rollout_was_catastrophic) {
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

        const double cvar = compute_empirical_lower_tail_cvar(sampled_returns, cvar_tau);
        return {mean, stddev, cvar, catastrophic_count};
    }

    static BettingGameMetrics evaluate_bettinggame_metrics(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double optimal_action_regret_threshold)
    {
        auto context = env->sample_context_itfc(env->get_initial_state_itfc());
        shared_ptr<const mcts::Action> recommended_action = root->recommend_action_itfc(*context);
        const int recommended_action_id =
            static_pointer_cast<const mcts::IntAction>(recommended_action)->action;

        BettingGameMetrics metrics{numeric_limits<double>::quiet_NaN(), 0.0};

        const auto action_cvar_it = root_solution.action_cvars.find(recommended_action_id);
        if (action_cvar_it != root_solution.action_cvars.end()) {
            metrics.cvar_regret = root_solution.optimal_cvar - action_cvar_it->second;
            if (metrics.cvar_regret <= optimal_action_regret_threshold + kCvarTolerance) {
                metrics.optimal_action_hit = 1.0;
            }
        }

        return metrics;
    }

    static vector<Candidate> build_candidates(double default_eval_tau) {
        vector<Candidate> cands;
        const vector<double> tau_values = {0.05, 0.1, 0.2, 0.25};

        for (double bias : {mcts::UctManagerArgs::USE_AUTO_BIAS, 2.0, 5.0, 10.0}) {
            for (double eps : {0.05, 0.1, 0.2, 0.5}) {
                string cfg = string("bias=") +
                    (bias == mcts::UctManagerArgs::USE_AUTO_BIAS ? "auto" : to_string(bias)) +
                    ",eps=" + to_string(eps);
                cands.push_back({"UCT", cfg, default_eval_tau, [bias, eps](auto env, auto init_state, int max_depth, int seed) {
                    mcts::UctManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.bias = bias;
                    args.epsilon_exploration = eps;
                    args.seed = seed;
                    auto mgr = make_shared<mcts::UctManager>(args);
                    auto root = make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
                    return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                }});
            }
        }

        for (int n_atoms : {25, 51, 100}) {
            for (double optimism : {2.0, 4.0, 8.0}) {
                for (double p_mean : {1.0, 2.0, 4.0}) {
                    for (double tau : tau_values) {
                        string cfg = "atoms=" + to_string(n_atoms)
                            + ",optimism=" + to_string(optimism)
                            + ",p=" + to_string(p_mean)
                            + ",tau=" + to_string(tau);
                        cands.push_back({"CATSO", cfg, tau, [n_atoms, optimism, p_mean, tau](auto env, auto init_state, int max_depth, int seed) {
                            mcts::CatsoManagerArgs args(env);
                            args.max_depth = max_depth;
                            args.mcts_mode = false;
                            args.n_atoms = n_atoms;
                            args.optimism_constant = optimism;
                            args.power_mean_exponent = p_mean;
                            args.cvar_tau = tau;
                            args.seed = seed;
                            auto mgr = make_shared<mcts::CatsoManager>(args);
                            auto root = make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
                            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                        }});
                    }
                }
            }
        }

        for (int max_particles : {32, 64, 128}) {
            for (double optimism : {2.0, 4.0, 8.0}) {
                for (double p_mean : {1.0, 2.0, 4.0}) {
                    for (double tau : tau_values) {
                        string cfg = "particles=" + to_string(max_particles)
                            + ",optimism=" + to_string(optimism)
                            + ",p=" + to_string(p_mean)
                            + ",tau=" + to_string(tau);
                        cands.push_back({"PATSO", cfg, tau, [max_particles, optimism, p_mean, tau](auto env, auto init_state, int max_depth, int seed) {
                            mcts::PatsoManagerArgs args(env);
                            args.max_depth = max_depth;
                            args.mcts_mode = false;
                            args.max_particles = max_particles;
                            args.optimism_constant = optimism;
                            args.power_mean_exponent = p_mean;
                            args.cvar_tau = tau;
                            args.seed = seed;
                            auto mgr = make_shared<mcts::PatsoManager>(args);
                            auto root = make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
                            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                        }});
                    }
                }
            }
        }

        return cands;
    }
}

int main(int argc, char** argv) {
    double optimal_action_regret_threshold = kDefaultOptimalActionRegretThreshold;
    for (int argi = 1; argi < argc; ++argi) {
        const string arg = argv[argi];
        if (arg == "--optimal-action-regret-threshold") {
            if (argi + 1 >= argc) {
                cerr << "Missing value for --optimal-action-regret-threshold\n";
                cerr << "Usage: " << argv[0]
                     << " [--optimal-action-regret-threshold <non-negative double>]\n";
                return 1;
            }

            const string value = argv[++argi];
            try {
                optimal_action_regret_threshold = stod(value);
            }
            catch (const exception&) {
                cerr << "Invalid value for --optimal-action-regret-threshold: " << value << "\n";
                return 1;
            }

            if (!isfinite(optimal_action_regret_threshold) || optimal_action_regret_threshold < 0.0) {
                cerr << "--optimal-action-regret-threshold must be a non-negative finite value\n";
                return 1;
            }
        }
        else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0]
                 << " [--optimal-action-regret-threshold <non-negative double>]\n";
            return 0;
        }
        else {
            cerr << "Unknown argument: " << arg << "\n";
            cerr << "Usage: " << argv[0]
                 << " [--optimal-action-regret-threshold <non-negative double>]\n";
            return 1;
        }
    }

    const double win_prob = 0.8;
    const int max_sequence_length = 6;
    const double max_state_value = mcts::exp::BettingGameEnv::default_max_state_value;
    const double initial_state = mcts::exp::BettingGameEnv::default_initial_state;
    const double reward_normalisation = mcts::exp::BettingGameEnv::default_reward_normalisation;
    const double cvar_tau = 0.2;
    const int horizon = max_sequence_length;
    const int total_trials = 30000;
    const int eval_rollouts = 200;
    const int runs = 2;
    const int threads = 16;
    const int base_seed = 4242;
    const int progress_batch_trials = min(total_trials, 10000);

    auto env = make_shared<mcts::exp::BettingGameEnv>(
        win_prob,
        max_sequence_length,
        max_state_value,
        initial_state,
        reward_normalisation);
    auto init_state = env->get_initial_state_itfc();
    auto candidates = build_candidates(cvar_tau);
    unordered_map<double, BettingGameCvarOracle> oracle_by_tau;
    for (const auto& cand : candidates) {
        oracle_by_tau.emplace(cand.eval_tau, BettingGameCvarOracle(env, cand.eval_tau));
    }
    const auto& base_root_solution = oracle_by_tau.at(cvar_tau).solve_state(env->get_initial_state());

    ofstream out("tune_bettinggame.csv", ios::out | ios::trunc);
    out << setprecision(17);
    out << "env,algorithm,config,eval_tau,run,mc_mean,mc_stddev,mc_cvar,cvar_regret,optimal_action_hit,catastrophic_count\n";

    map<string, map<double, unordered_map<string, Aggregate>>> agg;

    cout << "[tune] BettingGame"
         << ", horizon=" << horizon
         << ", trials=" << total_trials
         << ", default_eval_tau=" << cvar_tau
         << ", optimal_action_regret_threshold=" << optimal_action_regret_threshold
         << ", runs=" << runs
         << endl;
    cout << "  Root optimal CVaR (default_eval_tau): " << base_root_solution.optimal_cvar << endl;
    cout << "  Candidates: " << candidates.size()
         << ", total searches=" << (static_cast<long long>(candidates.size()) * runs)
         << ", total MCTS trials=" << (static_cast<long long>(candidates.size()) * runs * total_trials)
         << ", progress_batch_trials=" << progress_batch_trials
         << endl;

    const int run_summary_stride = max(1, runs / 10);
    for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
        const auto& cand = candidates[cand_idx];
        cout << "[" << (cand_idx + 1) << "/" << candidates.size() << "] "
             << cand.algo << " cfg=" << cand.config
             << ", eval_tau=" << cand.eval_tau << endl;
        for (int run = 0; run < runs; ++run) {
            int seed = base_seed + 1000 * run + static_cast<int>(hash<string>{}(cand.algo + cand.config) % 997);
            auto [mgr, root] = cand.make(env, init_state, horizon, seed);
            auto pool = make_unique<mcts::MctsPool>(mgr, root, threads);

            const bool print_trial_progress = (run == 0);
            if (print_trial_progress) {
                ostringstream label;
                label << "[" << (cand_idx + 1) << "/" << candidates.size() << "] "
                      << cand.algo << " run " << (run + 1) << "/" << runs;
                run_trials_with_progress(*pool, total_trials, progress_batch_trials, label.str());
            }
            else {
                pool->run_trials(total_trials, numeric_limits<double>::max(), true);
            }

            auto stats = evaluate_tree(env, root, horizon, eval_rollouts, threads, seed + 777, cand.eval_tau);
            const auto& root_solution = oracle_by_tau.at(cand.eval_tau).solve_state(env->get_initial_state());
            auto metrics = evaluate_bettinggame_metrics(
                env,
                root,
                root_solution,
                optimal_action_regret_threshold);
            out << "bettinggame," << cand.algo << "," << csv_escape(cand.config) << "," << cand.eval_tau << "," << run
                << "," << stats.mean << "," << stats.stddev
                << "," << stats.cvar
                << "," << metrics.cvar_regret << "," << metrics.optimal_action_hit
                << "," << stats.catastrophic_count << "\n";

            Aggregate& a = agg[cand.algo][cand.eval_tau][cand.config];
            a.count++;
            a.sum_mc_mean += stats.mean;
            a.sum_mc_stddev += stats.stddev;
            a.sum_mc_cvar += stats.cvar;
            a.sum_cvar_regret += metrics.cvar_regret;
            a.sum_optimal_action_hit += metrics.optimal_action_hit;
            a.sum_catastrophic_count += static_cast<double>(stats.catastrophic_count);

            if (run == 0 || run + 1 == runs || ((run + 1) % run_summary_stride == 0)) {
                cout << "    completed run " << (run + 1) << "/" << runs
                     << " regret=" << metrics.cvar_regret
                     << " hit=" << metrics.optimal_action_hit
                     << " mean=" << stats.mean
                     << endl;
            }
        }
        cout << "  done " << cand.algo << " cfg=" << cand.config << endl;
    }

    cout << "\nBest configs (avg over runs per config, minimizing CVaR regret, grouped by eval_tau):\n";
    cout << fixed << setprecision(6);
    for (const auto& [algo, tau_map] : agg) {
        for (const auto& [eval_tau, cfg_map] : tau_map) {
            double best_avg_regret = numeric_limits<double>::infinity();
            double best_avg_hit = -numeric_limits<double>::infinity();
            string best_cfg;
            for (const auto& [cfg, a] : cfg_map) {
                if (a.count <= 0) {
                    continue;
                }
                const double avg_regret = a.sum_cvar_regret / static_cast<double>(a.count);
                const double avg_hit = a.sum_optimal_action_hit / static_cast<double>(a.count);
                if (avg_regret < best_avg_regret - 1e-15 ||
                    (abs(avg_regret - best_avg_regret) <= 1e-15 && avg_hit > best_avg_hit))
                {
                    best_avg_regret = avg_regret;
                    best_avg_hit = avg_hit;
                    best_cfg = cfg;
                }
            }

            if (!best_cfg.empty()) {
                const Aggregate& a = cfg_map.at(best_cfg);
                cout << "  " << algo
                     << " (eval_tau=" << eval_tau << ")"
                     << " -> avg_regret=" << (a.sum_cvar_regret / static_cast<double>(a.count))
                     << ", avg_hit=" << (a.sum_optimal_action_hit / static_cast<double>(a.count))
                     << ", avg_mean=" << (a.sum_mc_mean / static_cast<double>(a.count))
                     << ", cfg=" << best_cfg << "\n";
            }
        }
    }

    cout << "Results written to tune_bettinggame.csv\n";
    return 0;
}
