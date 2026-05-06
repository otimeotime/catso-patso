// Single-config evaluator for Betting Game, intended to be driven by tune/tune.py.
//
// Accepts hyperparameters via CLI flags, runs MCTS for `--num-seeds` seeds at one
// trial budget, and emits ONE JSON line to stdout with mean/stddev metrics across
// seeds. Diagnostic logs go to stderr so stdout is parser-clean.
//
// This is a leaner, CLI-driven cousin of `tune_bettinggame.cpp` — it intentionally
// duplicates the env oracle and evaluator helpers rather than refactoring those
// into a shared header, because (a) those helpers are env-specific and (b) keeping
// the original tuner/runner untouched preserves paper reproducibility.

#include "env/betting_game_env.h"

#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include "exp/oracles/discrete_cvar_oracle_common.h"
#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

    // ---------------------------------------------------------------- CLI args

    struct CliArgs {
        string algo = "CATSO";  // UCT | CATSO | PATSO
        // Hyperparameters (subset relevant to chosen algo)
        int n_atoms = 51;            // CATSO
        int max_particles = 100;     // PATSO
        double optimism = 4.0;       // CATSO/PATSO
        double power_mean = 1.0;     // CATSO/PATSO
        double uct_bias = -1.0;      // UCT, -1 means "auto"
        double uct_epsilon = 0.1;    // UCT
        // Eval settings
        double cvar_tau = 0.25;
        double discount_gamma = 1.0;
        int trial_budget = 10000;
        int num_seeds = 3;
        int base_seed = 4242;
        int eval_rollouts = 100;
        int threads = 8;
        int horizon = 6;  // mirrors max_sequence_length default
        bool debug_trajectories = false;
        bool debug_root_actions = false;
        int debug_subtree_action = -1;
    };

    static void usage(ostream& out) {
        out << "Usage: mcts-eval-bettinggame --algo {UCT|CATSO|PATSO} [hyperparams] [eval]\n"
               "  Hyperparams (CATSO): --n-atoms INT --optimism FLOAT --power-mean FLOAT\n"
               "  Hyperparams (PATSO): --max-particles INT --optimism FLOAT --power-mean FLOAT\n"
               "  Hyperparams (UCT)  : --uct-bias FLOAT (or -1 for auto) --uct-epsilon FLOAT\n"
               "  Eval               : --cvar-tau FLOAT --gamma FLOAT --trial-budget INT --num-seeds INT\n"
               "                       --base-seed INT --eval-rollouts INT --threads INT --horizon INT\n"
               "                       [--debug-trajectories] [--debug-root-actions]\n"
               "                       [--debug-subtree-action INT]\n"
               "Output: ONE JSON line on stdout with averaged metrics.\n";
    }

    static bool parse_int(const string& s, int& out) {
        try { out = stoi(s); return true; } catch (...) { return false; }
    }
    static bool parse_double(const string& s, double& out) {
        try { out = stod(s); return true; } catch (...) { return false; }
    }

    static CliArgs parse_args(int argc, char** argv) {
        CliArgs args;
        for (int i = 1; i < argc; ++i) {
            string flag = argv[i];
            if (flag == "--help" || flag == "-h") {
                usage(cout);
                exit(0);
            }
            if (flag == "--debug-trajectories") {
                args.debug_trajectories = true;
                continue;
            }
            if (flag == "--debug-root-actions") {
                args.debug_root_actions = true;
                continue;
            }
            if (i + 1 >= argc) {
                cerr << "missing value for flag " << flag << "\n";
                usage(cerr);
                exit(2);
            }
            string val = argv[++i];

            if (flag == "--algo") args.algo = val;
            else if (flag == "--n-atoms") { if (!parse_int(val, args.n_atoms)) exit(2); }
            else if (flag == "--max-particles") { if (!parse_int(val, args.max_particles)) exit(2); }
            else if (flag == "--optimism") { if (!parse_double(val, args.optimism)) exit(2); }
            else if (flag == "--power-mean") { if (!parse_double(val, args.power_mean)) exit(2); }
            else if (flag == "--uct-bias") { if (!parse_double(val, args.uct_bias)) exit(2); }
            else if (flag == "--uct-epsilon") { if (!parse_double(val, args.uct_epsilon)) exit(2); }
            else if (flag == "--cvar-tau") { if (!parse_double(val, args.cvar_tau)) exit(2); }
            else if (flag == "--gamma") { if (!parse_double(val, args.discount_gamma)) exit(2); }
            else if (flag == "--trial-budget") { if (!parse_int(val, args.trial_budget)) exit(2); }
            else if (flag == "--num-seeds") { if (!parse_int(val, args.num_seeds)) exit(2); }
            else if (flag == "--base-seed") { if (!parse_int(val, args.base_seed)) exit(2); }
            else if (flag == "--eval-rollouts") { if (!parse_int(val, args.eval_rollouts)) exit(2); }
            else if (flag == "--threads") { if (!parse_int(val, args.threads)) exit(2); }
            else if (flag == "--horizon") { if (!parse_int(val, args.horizon)) exit(2); }
            else if (flag == "--debug-subtree-action") { if (!parse_int(val, args.debug_subtree_action)) exit(2); }
            else {
                cerr << "unknown flag: " << flag << "\n";
                usage(cerr);
                exit(2);
            }
        }

        if (args.algo != "UCT" && args.algo != "CATSO" && args.algo != "PATSO") {
            cerr << "--algo must be UCT, CATSO, or PATSO (got " << args.algo << ")\n";
            exit(2);
        }
        if (args.num_seeds <= 0 || args.trial_budget <= 0 || args.eval_rollouts <= 0) {
            cerr << "num-seeds, trial-budget, eval-rollouts must all be > 0\n";
            exit(2);
        }
        if (args.discount_gamma < 0.0 || args.discount_gamma > 1.0) {
            cerr << "gamma must be in [0,1]\n";
            exit(2);
        }
        return args;
    }

    // -------------------------------------------------------- discrete CVaR helpers

    using ReturnDistribution = map<double, double>;

    double compute_lower_tail_cvar(const ReturnDistribution& distribution, double tau) {
        if (distribution.empty()) return 0.0;
        const double tau_clamped = max(kCvarTolerance, min(tau, 1.0));
        double total_mass = 0.0;
        for (const auto& [v, p] : distribution) { (void)v; if (p > 0.0) total_mass += p; }
        if (total_mass <= 0.0) return 0.0;
        const double tail_mass = tau_clamped * total_mass;
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;
        for (const auto& [v, p] : distribution) {
            if (p <= 0.0) continue;
            const double remaining = max(0.0, tail_mass - cumulative_mass);
            const double take = min(p, remaining);
            tail_sum += take * v;
            cumulative_mass += p;
            if (cumulative_mass >= tail_mass) break;
        }
        return tail_sum / tail_mass;
    }

    double compute_empirical_lower_tail_cvar(vector<double> samples, double tau) {
        if (samples.empty()) return 0.0;
        sort(samples.begin(), samples.end());
        const double tau_clamped = max(kCvarTolerance, min(tau, 1.0));
        const double tail_mass = tau_clamped * static_cast<double>(samples.size());
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;
        for (double s : samples) {
            const double remaining = max(0.0, tail_mass - cumulative_mass);
            const double take = min(1.0, remaining);
            tail_sum += take * s;
            cumulative_mass += 1.0;
            if (cumulative_mass >= tail_mass) break;
        }
        return tail_sum / tail_mass;
    }

    // -------------------------------------------------------- BettingGame oracle

    struct BettingGameStateKey {
        double bankroll;
        int time;
        bool operator==(const BettingGameStateKey& o) const { return bankroll == o.bankroll && time == o.time; }
    };
    struct BettingGameStateKeyHash {
        size_t operator()(const BettingGameStateKey& k) const {
            size_t s = std::hash<double>{}(k.bankroll);
            s ^= std::hash<int>{}(k.time) + 0x9e3779b97f4a7c15ULL + (s << 6) + (s >> 2);
            return s;
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

    class BettingGameCvarOracle {
        private:
            shared_ptr<const mcts::exp::BettingGameEnv> env;
            double tau;
            unordered_map<BettingGameStateKey, OptimalDistributionSolution, BettingGameStateKeyHash> memo;

            static BettingGameStateKey make_key(shared_ptr<const mcts::exp::BettingGameState> state) {
                return {state->bankroll, state->time};
            }
            static void add_weighted_shifted_distribution(
                ReturnDistribution& out, const ReturnDistribution& in, double weight, double shift)
            {
                for (const auto& [v, p] : in) out[v + shift] += weight * p;
            }

        public:
            BettingGameCvarOracle(shared_ptr<const mcts::exp::BettingGameEnv> env, double tau)
                : env(move(env)), tau(tau) {}

            const OptimalDistributionSolution& solve_state(shared_ptr<const mcts::exp::BettingGameState> state) {
                const BettingGameStateKey key = make_key(state);
                const auto it = memo.find(key);
                if (it != memo.end()) return it->second;

                OptimalDistributionSolution solution;
                if (env->is_sink_state(state)) {
                    solution.state_value_distribution[0.0] = 1.0;
                    auto [ins, _] = memo.emplace(key, move(solution));
                    return ins->second;
                }

                auto actions = env->get_valid_actions(state);
                for (const auto& action : *actions) {
                    ReturnDistribution action_distribution;
                    auto transitions = env->get_transition_distribution(state, action);
                    for (const auto& [next_state, probability] : *transitions) {
                        const auto& child = solve_state(next_state);
                        const double r = env->get_reward(state, action, next_state);
                        add_weighted_shifted_distribution(
                            action_distribution, child.state_value_distribution, probability, r);
                    }
                    const int aid = action->action;
                    const double action_cvar = compute_lower_tail_cvar(action_distribution, tau);
                    solution.action_value_distributions.emplace(aid, action_distribution);
                    solution.action_cvars.emplace(aid, action_cvar);
                    if (solution.optimal_actions.empty() || action_cvar > solution.optimal_cvar + kCvarTolerance) {
                        solution.optimal_cvar = action_cvar;
                        solution.optimal_actions = {aid};
                        solution.canonical_action = aid;
                    } else if (abs(action_cvar - solution.optimal_cvar) <= kCvarTolerance) {
                        solution.optimal_actions.push_back(aid);
                        solution.canonical_action = min(solution.canonical_action, aid);
                    }
                }
                if (solution.canonical_action >= 0) {
                    solution.state_value_distribution = solution.action_value_distributions.at(solution.canonical_action);
                }
                auto [ins, _] = memo.emplace(key, move(solution));
                return ins->second;
            }
    };

    // ---------------------------------------------------- Per-seed eval pipeline

    struct PerSeedMetrics {
        double mc_mean = 0.0;
        double mc_stddev = 0.0;
        double mc_cvar = 0.0;
        double cvar_regret = numeric_limits<double>::quiet_NaN();
        double optimal_action_hit = 0.0;
        int catastrophic_count = 0;
        double wall_time_sec = 0.0;
    };

    static PerSeedMetrics evaluate_tree(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        int horizon,
        int eval_rollouts,
        int seed,
        double cvar_tau,
        bool debug_trajectories,
        const string& debug_trajectory_label)
    {
        PerSeedMetrics m;
        mcts::RandManager eval_rng(seed);
        mcts::EvalPolicy policy(root, env, eval_rng);
        vector<double> sampled_returns;
        sampled_returns.reserve(static_cast<size_t>(eval_rollouts));
        int catastrophic_count = 0;
        const bool should_debug_trajectories =
            debug_trajectories || mcts::should_debug_mc_eval_trajectories_from_env();
        const string trajectory_label = debug_trajectory_label.empty() ? "MC" : debug_trajectory_label;

        for (int rollout = 0; rollout < eval_rollouts; ++rollout) {
            policy.reset();
            int steps = 0;
            double sample_return = 0.0;
            shared_ptr<const mcts::State> state = env->get_initial_state_itfc();
            bool was_catastrophic = env->is_catastrophic_state_itfc(state);
            auto context_ptr = env->sample_context_itfc(state);
            mcts::MctsEnvContext& context = *context_ptr;
            mcts::RolloutTrajectoryTrace trajectory_trace(should_debug_trajectories, trajectory_label);
            trajectory_trace.append_state(state);

            while (steps < horizon && !env->is_sink_state_itfc(state)) {
                auto action = policy.get_action(state, context);
                auto next_state = env->sample_transition_distribution_itfc(state, action, eval_rng);
                auto observation = env->sample_observation_distribution_itfc(action, next_state, eval_rng);
                const double reward = env->get_reward_itfc(state, action, observation);
                sample_return += reward;
                trajectory_trace.append_reward(reward);
                trajectory_trace.append_state(next_state);
                if (env->is_catastrophic_state_itfc(next_state)) was_catastrophic = true;
                policy.update_step(action, observation);
                state = next_state;
                ++steps;
            }
            trajectory_trace.print(cerr);
            sampled_returns.push_back(sample_return);
            if (was_catastrophic) ++catastrophic_count;
        }

        if (sampled_returns.empty()) return m;

        double sum = 0.0;
        for (double r : sampled_returns) sum += r;
        m.mc_mean = sum / static_cast<double>(sampled_returns.size());
        if (sampled_returns.size() > 1u) {
            double var = 0.0;
            for (double r : sampled_returns) var += pow(r - m.mc_mean, 2.0);
            var /= static_cast<double>(sampled_returns.size() - 1u);
            m.mc_stddev = sqrt(var);
        }
        m.mc_cvar = compute_empirical_lower_tail_cvar(sampled_returns, cvar_tau);
        m.catastrophic_count = catastrophic_count;
        return m;
    }

    static void evaluate_root_recommendation(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double eval_tau,
        PerSeedMetrics& m)
    {
        auto context = env->sample_context_itfc(env->get_initial_state_itfc());
        auto recommended = root->recommend_action_itfc(*context);
        const int rid = static_pointer_cast<const mcts::IntAction>(recommended)->action;
        const auto it = root_solution.action_cvars.find(rid);
        if (it != root_solution.action_cvars.end()) {
            const double estimated_action_cvar =
                mcts::exp::oracles::estimate_recommended_action_cvar(root, recommended, eval_tau);
            if (!std::isnan(estimated_action_cvar)) {
                m.cvar_regret = std::abs(root_solution.optimal_cvar - estimated_action_cvar);
                if (m.cvar_regret <= kCvarTolerance) m.optimal_action_hit = 1.0;
            }
        }
    }

    static double compute_bet_amount(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::exp::BettingGameState> state,
        int action_id)
    {
        const double action_fraction =
            static_cast<double>(action_id) / static_cast<double>(mcts::exp::BettingGameEnv::num_actions - 1);
        return min(state->bankroll * action_fraction, env->get_max_state_value() - state->bankroll);
    }

    static string format_action_ids(const vector<int>& action_ids) {
        ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < action_ids.size(); ++i) {
            if (i > 0) oss << ",";
            oss << action_ids[i];
        }
        oss << "]";
        return oss.str();
    }

    static string format_optional_double(double value) {
        if (std::isnan(value)) {
            return "na";
        }
        ostringstream oss;
        oss << setprecision(10) << value;
        return oss.str();
    }

    static string format_optional_int(int value) {
        if (value < 0) {
            return "na";
        }
        return to_string(value);
    }

    static bool same_betting_game_state(
        shared_ptr<const mcts::exp::BettingGameState> lhs,
        shared_ptr<const mcts::exp::BettingGameState> rhs)
    {
        return lhs != nullptr
            && rhs != nullptr
            && lhs->bankroll == rhs->bankroll
            && lhs->time == rhs->time;
    }

    static void print_root_action_debug(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double eval_tau,
        int seed_index)
    {
        auto context = env->sample_context_itfc(env->get_initial_state_itfc());
        auto recommended = root->recommend_action_itfc(*context);
        const int recommended_action_id =
            static_pointer_cast<const mcts::IntAction>(recommended)->action;
        const auto root_state = env->get_initial_state();

        unordered_map<int, shared_ptr<const mcts::Action>> expanded_actions;
        unordered_map<int, int> expanded_visits;
        for (const auto& action_ptr : root->get_child_actions_itfc()) {
            const int action_id = static_pointer_cast<const mcts::IntAction>(action_ptr)->action;
            expanded_actions[action_id] = action_ptr;
            expanded_visits[action_id] = root->get_child_node_itfc(action_ptr)->get_num_visits();
        }

        cerr << "[debug-root] seed=" << seed_index
             << " recommended_action=" << recommended_action_id
             << " bet=" << compute_bet_amount(env, root_state, recommended_action_id)
             << " oracle_optimal_cvar=" << root_solution.optimal_cvar
             << " oracle_optimal_actions=" << format_action_ids(root_solution.optimal_actions)
             << " root_visits=" << root->get_num_visits()
             << " expanded_children=" << root->get_num_children()
             << "\n";
        cerr << "[debug-root] action_id bet visits oracle_cvar estimated_edge_cvar tags\n";

        const auto legal_actions = env->get_valid_actions(root_state);
        for (const auto& legal_action : *legal_actions) {
            const int action_id = legal_action->action;
            const double bet = compute_bet_amount(env, root_state, action_id);
            const auto oracle_it = root_solution.action_cvars.find(action_id);
            const double oracle_cvar =
                (oracle_it == root_solution.action_cvars.end())
                    ? numeric_limits<double>::quiet_NaN()
                    : oracle_it->second;

            double estimated_action_cvar = numeric_limits<double>::quiet_NaN();
            int visits = 0;
            if (const auto expanded_it = expanded_actions.find(action_id); expanded_it != expanded_actions.end()) {
                visits = expanded_visits[action_id];
                estimated_action_cvar =
                    mcts::exp::oracles::estimate_recommended_action_cvar(root, expanded_it->second, eval_tau);
            }

            string tags;
            if (action_id == recommended_action_id) {
                tags += "recommended";
            }
            if (find(root_solution.optimal_actions.begin(), root_solution.optimal_actions.end(), action_id) !=
                root_solution.optimal_actions.end()) {
                if (!tags.empty()) tags += ",";
                tags += "oracle-optimal";
            }

            cerr << "[debug-root] "
                 << action_id << " "
                 << bet << " "
                 << visits << " "
                 << format_optional_double(oracle_cvar) << " "
                 << format_optional_double(estimated_action_cvar) << " "
                 << (tags.empty() ? "-" : tags)
                 << "\n";
        }
    }

    static void print_root_action_subtree_debug(
        shared_ptr<const mcts::exp::BettingGameEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        BettingGameCvarOracle& oracle,
        double eval_tau,
        double discount_gamma,
        int action_id,
        int seed_index)
    {
        shared_ptr<const mcts::Action> action_ptr = nullptr;
        for (const auto& child_action : root->get_child_actions_itfc()) {
            const int child_action_id = static_pointer_cast<const mcts::IntAction>(child_action)->action;
            if (child_action_id == action_id) {
                action_ptr = child_action;
                break;
            }
        }

        if (action_ptr == nullptr) {
            cerr << "[debug-subtree] seed=" << seed_index
                 << " action_id=" << action_id
                 << " status=not-expanded\n";
            return;
        }

        const auto chance_node =
            dynamic_pointer_cast<const mcts::CatsoCNode>(root->get_child_node_itfc(action_ptr));
        if (chance_node == nullptr) {
            cerr << "[debug-subtree] seed=" << seed_index
                 << " action_id=" << action_id
                 << " status=unsupported-root-algorithm\n";
            return;
        }

        const auto root_state = env->get_initial_state();
        const auto typed_action = static_pointer_cast<const mcts::IntAction>(action_ptr);
        const auto transitions = env->get_transition_distribution(root_state, typed_action);
        const auto expanded_observations = chance_node->get_child_observations_itfc();
        ReturnDistribution snapshot_distribution;
        double snapshot_mean = 0.0;

        const auto oracle_action_it = root_solution.action_cvars.find(action_id);
        const double oracle_action_cvar =
            oracle_action_it == root_solution.action_cvars.end()
                ? numeric_limits<double>::quiet_NaN()
                : oracle_action_it->second;

        cerr << "[debug-subtree] seed=" << seed_index
             << " root_action=" << action_id
             << " bet=" << compute_bet_amount(env, root_state, action_id)
             << " chance_visits=" << chance_node->get_num_visits()
             << " chance_mean=" << chance_node->get_mean_value()
             << " chance_cvar=" << chance_node->get_cvar_value_at(eval_tau)
             << " oracle_action_cvar=" << format_optional_double(oracle_action_cvar)
             << " gamma=" << discount_gamma
             << "\n";
        cerr << "[debug-subtree] child_state prob immediate_reward child_value gamma_child_value "
                "backed_up_q oracle_state_cvar child_visits child_rec_action child_rec_edge_cvar\n";

        for (const auto& [next_state_raw, probability] : *transitions) {
            const auto next_state = static_pointer_cast<const mcts::exp::BettingGameState>(next_state_raw);
            const double immediate_reward = env->get_reward(root_state, typed_action, next_state);
            shared_ptr<const mcts::exp::BettingGameState> expanded_state = nullptr;

            for (const auto& observation : expanded_observations) {
                const auto observed_state = static_pointer_cast<const mcts::exp::BettingGameState>(observation);
                if (same_betting_game_state(observed_state, next_state)) {
                    expanded_state = observed_state;
                    break;
                }
            }

            double child_value = 0.0;
            int child_visits = 0;
            int child_recommended_action_id = -1;
            double child_recommended_edge_cvar = numeric_limits<double>::quiet_NaN();
            if (expanded_state != nullptr && chance_node->has_child_node(expanded_state)) {
                const auto child_node =
                    dynamic_pointer_cast<const mcts::CatsoDNode>(chance_node->get_child_node(expanded_state));
                if (child_node != nullptr) {
                    child_value = child_node->get_value_estimate();
                    child_visits = child_node->get_num_visits();
                    auto child_context =
                        env->sample_context_itfc(static_pointer_cast<const mcts::State>(expanded_state));
                    auto child_recommended_action = child_node->recommend_action_itfc(*child_context);
                    child_recommended_action_id =
                        static_pointer_cast<const mcts::IntAction>(child_recommended_action)->action;
                    child_recommended_edge_cvar = mcts::exp::oracles::estimate_recommended_action_cvar(
                        static_pointer_cast<const mcts::MctsDNode>(child_node),
                        child_recommended_action,
                        eval_tau);
                }
            }

            const double backed_up_q = immediate_reward + discount_gamma * child_value;
            snapshot_distribution[backed_up_q] += probability;
            snapshot_mean += probability * backed_up_q;

            const auto& child_solution = oracle.solve_state(next_state);
            cerr << "[debug-subtree] "
                 << next_state->get_pretty_print_string() << " "
                 << probability << " "
                 << immediate_reward << " "
                 << child_value << " "
                 << (discount_gamma * child_value) << " "
                 << backed_up_q << " "
                 << child_solution.optimal_cvar << " "
                 << child_visits << " "
                 << format_optional_int(child_recommended_action_id) << " "
                 << format_optional_double(child_recommended_edge_cvar)
                 << "\n";

            if (expanded_state != nullptr && chance_node->has_child_node(expanded_state)) {
                const auto child_node =
                    dynamic_pointer_cast<const mcts::CatsoDNode>(chance_node->get_child_node(expanded_state));
                if (child_node != nullptr) {
                    unordered_map<int, shared_ptr<const mcts::Action>> expanded_actions;
                    unordered_map<int, int> expanded_visits;
                    for (const auto& child_action : child_node->get_child_actions_itfc()) {
                        const int child_action_id =
                            static_pointer_cast<const mcts::IntAction>(child_action)->action;
                        expanded_actions[child_action_id] = child_action;
                        expanded_visits[child_action_id] =
                            child_node->get_child_node_itfc(child_action)->get_num_visits();
                    }

                    cerr << "[debug-subtree-actions] state=" << next_state->get_pretty_print_string()
                         << " value_estimate=" << child_node->get_value_estimate()
                         << " recommended_action=" << format_optional_int(child_recommended_action_id)
                         << "\n";
                    cerr << "[debug-subtree-actions] action_id bet visits oracle_cvar estimated_edge_cvar tags\n";

                    const auto legal_actions = env->get_valid_actions(next_state);
                    for (const auto& legal_action : *legal_actions) {
                        const int legal_action_id = legal_action->action;
                        const double legal_bet = compute_bet_amount(env, next_state, legal_action_id);
                        const auto oracle_child_it = child_solution.action_cvars.find(legal_action_id);
                        const double oracle_child_cvar =
                            oracle_child_it == child_solution.action_cvars.end()
                                ? numeric_limits<double>::quiet_NaN()
                                : oracle_child_it->second;
                        double estimated_child_edge_cvar = numeric_limits<double>::quiet_NaN();
                        int action_visits = 0;
                        if (const auto expanded_it = expanded_actions.find(legal_action_id);
                            expanded_it != expanded_actions.end())
                        {
                            action_visits = expanded_visits[legal_action_id];
                            estimated_child_edge_cvar = mcts::exp::oracles::estimate_recommended_action_cvar(
                                static_pointer_cast<const mcts::MctsDNode>(child_node),
                                expanded_it->second,
                                eval_tau);
                        }

                        string tags;
                        if (legal_action_id == child_recommended_action_id) {
                            tags += "recommended";
                        }
                        if (find(
                                child_solution.optimal_actions.begin(),
                                child_solution.optimal_actions.end(),
                                legal_action_id) != child_solution.optimal_actions.end())
                        {
                            if (!tags.empty()) tags += ",";
                            tags += "oracle-optimal";
                        }

                        cerr << "[debug-subtree-actions] "
                             << legal_action_id << " "
                             << legal_bet << " "
                             << action_visits << " "
                             << format_optional_double(oracle_child_cvar) << " "
                             << format_optional_double(estimated_child_edge_cvar) << " "
                             << (tags.empty() ? "-" : tags)
                             << "\n";
                    }
                }
            }
        }

        const double snapshot_cvar = compute_lower_tail_cvar(snapshot_distribution, eval_tau);
        cerr << "[debug-subtree] snapshot_mean=" << snapshot_mean
             << " snapshot_cvar=" << snapshot_cvar
             << " chance_minus_snapshot=" << (chance_node->get_cvar_value_at(eval_tau) - snapshot_cvar)
             << "\n";
    }

    // ------------------------------------------------- Manager / root construction

    using MgrAndRoot = pair<shared_ptr<mcts::MctsManager>, shared_ptr<mcts::MctsDNode>>;

    static MgrAndRoot build_planner(
        const CliArgs& args,
        shared_ptr<mcts::MctsEnv> env,
        shared_ptr<const mcts::State> init_state,
        int seed)
    {
        const int max_depth = args.horizon;
        if (args.algo == "UCT") {
            mcts::UctManagerArgs a(env);
            a.max_depth = max_depth;
            a.mcts_mode = false;
            a.bias = (args.uct_bias < 0.0) ? mcts::UctManagerArgs::USE_AUTO_BIAS : args.uct_bias;
            a.epsilon_exploration = args.uct_epsilon;
            a.seed = seed;
            auto mgr = make_shared<mcts::UctManager>(a);
            auto root = make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
            return {static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root)};
        }
        if (args.algo == "CATSO") {
            mcts::CatsoManagerArgs a(env);
            a.max_depth = max_depth;
            a.mcts_mode = false;
            a.n_atoms = args.n_atoms;
            a.optimism_constant = args.optimism;
            a.power_mean_exponent = args.power_mean;
            a.cvar_tau = args.cvar_tau;
            a.discount_gamma = args.discount_gamma;
            a.seed = seed;
            auto mgr = make_shared<mcts::CatsoManager>(a);
            auto root = make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
            return {static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root)};
        }
        // PATSO
        mcts::PatsoManagerArgs a(env);
        a.max_depth = max_depth;
        a.mcts_mode = false;
        a.max_particles = args.max_particles;
        a.optimism_constant = args.optimism;
        a.power_mean_exponent = args.power_mean;
        a.cvar_tau = args.cvar_tau;
        a.discount_gamma = args.discount_gamma;
        a.seed = seed;
        auto mgr = make_shared<mcts::PatsoManager>(a);
        auto root = make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
        return {static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root)};
    }

    // ------------------------------------------------- JSON output (hand-built)

    static string json_double(double v) {
        if (std::isnan(v)) return "null";
        ostringstream oss;
        oss << setprecision(17) << v;
        return oss.str();
    }

    struct AggregatedMetrics {
        int n = 0;
        int regret_n = 0;
        double sum_mc_mean = 0.0;
        double sum_mc_stddev = 0.0;
        double sum_mc_cvar = 0.0;
        double sum_cvar_regret = 0.0;
        double sum_opt_hit = 0.0;
        double sum_catastrophic = 0.0;
        double sum_wall_time = 0.0;

        // For per-seed stddev of cvar_regret (the noise we hand to Optuna)
        vector<double> regrets;

        void add(const PerSeedMetrics& m) {
            ++n;
            sum_mc_mean += m.mc_mean;
            sum_mc_stddev += m.mc_stddev;
            sum_mc_cvar += m.mc_cvar;
            sum_opt_hit += m.optimal_action_hit;
            sum_catastrophic += static_cast<double>(m.catastrophic_count);
            sum_wall_time += m.wall_time_sec;
            if (!std::isnan(m.cvar_regret)) {
                ++regret_n;
                sum_cvar_regret += m.cvar_regret;
                regrets.push_back(m.cvar_regret);
            }
        }

        double mean(double s) const { return n > 0 ? s / static_cast<double>(n) : 0.0; }
        double mean_regret() const {
            return regret_n > 0 ? sum_cvar_regret / static_cast<double>(regret_n)
                                : numeric_limits<double>::quiet_NaN();
        }
        double regret_stddev() const {
            if (regret_n < 2) return 0.0;
            const double mr = mean_regret();
            double var = 0.0;
            for (double r : regrets) var += (r - mr) * (r - mr);
            return sqrt(var / static_cast<double>(regret_n - 1));
        }
    };
}  // namespace

int main(int argc, char** argv) {
    const CliArgs args = parse_args(argc, argv);

    // Use the env's published defaults so this binary stays in sync with run_bettinggame.
    const double win_prob = 0.8;
    const int max_sequence_length = args.horizon;  // Betting Game defines horizon == max_sequence_length
    const double max_state_value = mcts::exp::BettingGameEnv::default_max_state_value;
    const double initial_state = mcts::exp::BettingGameEnv::default_initial_state;
    const double reward_normalisation = max_state_value;

    auto env = make_shared<mcts::exp::BettingGameEnv>(
        win_prob, max_sequence_length, max_state_value, initial_state, reward_normalisation);

    // Diagnostics to stderr (stdout is reserved for the JSON line).
    cerr << "[eval-bettinggame] algo=" << args.algo
         << ", cvar_tau=" << args.cvar_tau
         << ", gamma=" << args.discount_gamma
         << ", trial_budget=" << args.trial_budget
         << ", num_seeds=" << args.num_seeds
         << ", eval_rollouts=" << args.eval_rollouts
         << ", threads=" << args.threads
         << ", horizon=" << args.horizon
         << "\n";

    BettingGameCvarOracle oracle(env, args.cvar_tau);
    const auto& root_solution = oracle.solve_state(env->get_initial_state());
    cerr << "[eval-bettinggame] root_optimal_cvar=" << root_solution.optimal_cvar << "\n";

    auto init_state = env->get_initial_state_itfc();
    AggregatedMetrics agg;

    for (int s = 0; s < args.num_seeds; ++s) {
        const int seed = args.base_seed
            + 1000 * s
            + static_cast<int>(hash<string>{}(args.algo) % 997);
        const auto t0 = chrono::steady_clock::now();

        auto [mgr, root] = build_planner(args, static_pointer_cast<mcts::MctsEnv>(env), init_state, seed);
        auto pool = make_unique<mcts::MctsPool>(mgr, root, args.threads);
        pool->run_trials(args.trial_budget, numeric_limits<double>::max(), true);

        PerSeedMetrics m = evaluate_tree(
            env,
            root,
            args.horizon,
            args.eval_rollouts,
            seed + 777,
            args.cvar_tau,
            args.debug_trajectories,
            args.algo);
        evaluate_root_recommendation(env, root, root_solution, args.cvar_tau, m);
        if (args.debug_root_actions) {
            print_root_action_debug(env, root, root_solution, args.cvar_tau, s + 1);
        }
        if (args.debug_subtree_action >= 0) {
            print_root_action_subtree_debug(
                env,
                root,
                root_solution,
                oracle,
                args.cvar_tau,
                args.discount_gamma,
                args.debug_subtree_action,
                s + 1);
        }
        m.wall_time_sec =
            chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - t0).count();

        cerr << "[eval-bettinggame] seed " << (s + 1) << "/" << args.num_seeds
             << " regret=" << m.cvar_regret
             << " mc_cvar=" << m.mc_cvar
             << " mc_mean=" << m.mc_mean
             << " (" << fixed << setprecision(2) << m.wall_time_sec << "s)\n";

        agg.add(m);
    }

    // Single-line JSON to stdout. Keys:
    //   cvar_regret        — primary objective (lower is better)
    //   cvar_regret_stddev — across-seed dispersion (helps Optuna noise model later)
    //   mc_mean / mc_cvar / optimal_action_prob / catastrophic_count — diagnostics
    //   num_seeds          — sanity
    //   wall_time_sec      — total per-binary wall-clock
    cout << "{"
         << "\"cvar_regret\":" << json_double(agg.mean_regret())
         << ",\"cvar_regret_stddev\":" << json_double(agg.regret_stddev())
         << ",\"mc_mean\":" << json_double(agg.mean(agg.sum_mc_mean))
         << ",\"mc_cvar\":" << json_double(agg.mean(agg.sum_mc_cvar))
         << ",\"mc_stddev\":" << json_double(agg.mean(agg.sum_mc_stddev))
         << ",\"optimal_action_prob\":" << json_double(agg.mean(agg.sum_opt_hit))
         << ",\"catastrophic_count\":" << json_double(agg.mean(agg.sum_catastrophic))
         << ",\"num_seeds\":" << agg.n
         << ",\"wall_time_sec\":" << json_double(agg.sum_wall_time)
         << "}" << endl;
    return 0;
}
