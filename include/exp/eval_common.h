#pragma once
//
// Shared infrastructure for single-config evaluators (mcts-eval-<env>) driven
// by tune/tune.py. Each per-env eval binary instantiates these templates with
// its own env type, oracle, and catastrophe predicate; the env-agnostic plumbing
// (CLI parser, CVaR helpers, evaluate-tree loop, JSON output) lives here.
//
// Not used by the existing run_<env>.cpp / tune_<env>.cpp binaries — those
// remain as-is for paper reproducibility.

#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/uct/max_uct_decision_node.h"
#include "algorithms/uct/power_uct_decision_node.h"
#include "algorithms/uct/power_uct_manager.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mcts::exp::eval {

constexpr double kCvarTolerance = 1e-12;

// ------------------------------------------------------------------ CLI args

struct CliArgs {
    std::string algo = "CATSO";  // UCT | MaxUCT | PowerUCT | CATSO | PATSO

    // CATSO
    int n_atoms = 51;
    // PATSO
    int max_particles = 100;
    // CATSO + PATSO
    double optimism = 4.0;
    double power_mean = 1.0;
    // UCT
    double uct_bias = -1.0;       // -1 means "auto"
    double uct_epsilon = 0.1;

    // Eval settings
    double cvar_tau = 0.1;
    double discount_gamma = 1.0;
    int trial_budget = 10000;
    int num_seeds = 3;
    int base_seed = 4242;
    int eval_rollouts = 100;
    int threads = 8;
    int horizon = 0;              // env-specific; required.
    bool debug_trajectories = false;
};

inline void usage(std::ostream& out, const std::string& binary_name) {
    out << "Usage: " << binary_name
        << " --algo {UCT|MaxUCT|PowerUCT|CATSO|PATSO} [hyperparams] [eval]\n"
           "  Hyperparams (CATSO): --n-atoms INT --optimism FLOAT --power-mean FLOAT\n"
           "  Hyperparams (PATSO): --max-particles INT --optimism FLOAT --power-mean FLOAT\n"
           "  Hyperparams (UCT/*): --uct-bias FLOAT (-1=auto) --uct-epsilon FLOAT\n"
           "  Hyperparams (PowerUCT): --power-mean FLOAT\n"
           "  Eval               : --cvar-tau FLOAT --gamma FLOAT --trial-budget INT --num-seeds INT\n"
           "                       --base-seed INT --eval-rollouts INT --threads INT --horizon INT\n"
           "                       [--debug-trajectories]\n"
           "Output: ONE JSON line on stdout with averaged metrics.\n";
}

inline bool parse_int(const std::string& s, int& out) {
    try { out = std::stoi(s); return true; } catch (...) { return false; }
}
inline bool parse_double(const std::string& s, double& out) {
    try { out = std::stod(s); return true; } catch (...) { return false; }
}

inline CliArgs parse_args(
    int argc,
    char** argv,
    int default_horizon,
    double default_discount_gamma = 1.0)
{
    CliArgs args;
    args.horizon = default_horizon;
    args.discount_gamma = default_discount_gamma;
    const std::string binary_name = argc > 0 ? argv[0] : "mcts-eval-<env>";

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") { usage(std::cout, binary_name); std::exit(0); }
        if (flag == "--debug-trajectories") {
            args.debug_trajectories = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "missing value for flag " << flag << "\n";
            usage(std::cerr, binary_name);
            std::exit(2);
        }
        std::string val = argv[++i];

        if (flag == "--algo") args.algo = val;
        else if (flag == "--n-atoms")        { if (!parse_int(val, args.n_atoms)) std::exit(2); }
        else if (flag == "--max-particles")  { if (!parse_int(val, args.max_particles)) std::exit(2); }
        else if (flag == "--optimism")       { if (!parse_double(val, args.optimism)) std::exit(2); }
        else if (flag == "--power-mean")     { if (!parse_double(val, args.power_mean)) std::exit(2); }
        else if (flag == "--uct-bias")       { if (!parse_double(val, args.uct_bias)) std::exit(2); }
        else if (flag == "--uct-epsilon")    { if (!parse_double(val, args.uct_epsilon)) std::exit(2); }
        else if (flag == "--cvar-tau")       { if (!parse_double(val, args.cvar_tau)) std::exit(2); }
        else if (flag == "--gamma")          { if (!parse_double(val, args.discount_gamma)) std::exit(2); }
        else if (flag == "--trial-budget")   { if (!parse_int(val, args.trial_budget)) std::exit(2); }
        else if (flag == "--num-seeds")      { if (!parse_int(val, args.num_seeds)) std::exit(2); }
        else if (flag == "--base-seed")      { if (!parse_int(val, args.base_seed)) std::exit(2); }
        else if (flag == "--eval-rollouts")  { if (!parse_int(val, args.eval_rollouts)) std::exit(2); }
        else if (flag == "--threads")        { if (!parse_int(val, args.threads)) std::exit(2); }
        else if (flag == "--horizon")        { if (!parse_int(val, args.horizon)) std::exit(2); }
        else {
            std::cerr << "unknown flag: " << flag << "\n";
            usage(std::cerr, binary_name);
            std::exit(2);
        }
    }

    if (args.algo != "UCT"
        && args.algo != "MaxUCT"
        && args.algo != "PowerUCT"
        && args.algo != "CATSO"
        && args.algo != "PATSO")
    {
        std::cerr << "--algo must be UCT, MaxUCT, PowerUCT, CATSO, or PATSO (got "
                  << args.algo << ")\n";
        std::exit(2);
    }
    if (args.num_seeds <= 0 || args.trial_budget <= 0 || args.eval_rollouts <= 0
        || args.horizon <= 0)
    {
        std::cerr << "num-seeds, trial-budget, eval-rollouts, horizon must all be > 0\n";
        std::exit(2);
    }
    if (args.discount_gamma < 0.0 || args.discount_gamma > 1.0) {
        std::cerr << "gamma must be in [0,1]\n";
        std::exit(2);
    }
    return args;
}

// -------------------------------------------------------- discrete CVaR helpers

using ReturnDistribution = std::map<double, double>;

inline double compute_lower_tail_cvar(const ReturnDistribution& distribution, double tau) {
    if (distribution.empty()) return 0.0;
    const double tau_clamped = std::max(kCvarTolerance, std::min(tau, 1.0));
    double total_mass = 0.0;
    for (const auto& [v, p] : distribution) { (void)v; if (p > 0.0) total_mass += p; }
    if (total_mass <= 0.0) return 0.0;
    const double tail_mass = tau_clamped * total_mass;
    double cumulative_mass = 0.0;
    double tail_sum = 0.0;
    for (const auto& [v, p] : distribution) {
        if (p <= 0.0) continue;
        const double remaining = std::max(0.0, tail_mass - cumulative_mass);
        const double take = std::min(p, remaining);
        tail_sum += take * v;
        cumulative_mass += p;
        if (cumulative_mass >= tail_mass) break;
    }
    return tail_sum / tail_mass;
}

inline double compute_empirical_lower_tail_cvar(std::vector<double> samples, double tau) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const double tau_clamped = std::max(kCvarTolerance, std::min(tau, 1.0));
    const double tail_mass = tau_clamped * static_cast<double>(samples.size());
    double cumulative_mass = 0.0;
    double tail_sum = 0.0;
    for (double s : samples) {
        const double remaining = std::max(0.0, tail_mass - cumulative_mass);
        const double take = std::min(1.0, remaining);
        tail_sum += take * s;
        cumulative_mass += 1.0;
        if (cumulative_mass >= tail_mass) break;
    }
    return tail_sum / tail_mass;
}

// ----------------------------------------------------- per-seed metrics + agg

struct PerSeedMetrics {
    double mc_mean = 0.0;
    double mc_stddev = 0.0;
    double mc_cvar = 0.0;
    double cvar_regret = std::numeric_limits<double>::quiet_NaN();
    double optimal_action_hit = 0.0;
    int catastrophic_count = 0;
    double wall_time_sec = 0.0;
};

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
    std::vector<double> regrets;

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
                            : std::numeric_limits<double>::quiet_NaN();
    }
    double regret_stddev() const {
        if (regret_n < 2) return 0.0;
        const double mr = mean_regret();
        double var = 0.0;
        for (double r : regrets) var += (r - mr) * (r - mr);
        return std::sqrt(var / static_cast<double>(regret_n - 1));
    }
};

// ------------------------------------------------- planner / root construction

using MgrAndRoot = std::pair<std::shared_ptr<mcts::MctsManager>, std::shared_ptr<mcts::MctsDNode>>;

inline MgrAndRoot build_planner(
    const CliArgs& args,
    std::shared_ptr<mcts::MctsEnv> env,
    std::shared_ptr<const mcts::State> init_state,
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
        auto mgr = std::make_shared<mcts::UctManager>(a);
        auto root = std::make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
        return {std::static_pointer_cast<mcts::MctsManager>(mgr),
                std::static_pointer_cast<mcts::MctsDNode>(root)};
    }
    if (args.algo == "MaxUCT") {
        mcts::UctManagerArgs a(env);
        a.max_depth = max_depth;
        a.mcts_mode = false;
        a.bias = (args.uct_bias < 0.0) ? mcts::UctManagerArgs::USE_AUTO_BIAS : args.uct_bias;
        a.recommend_most_visited = false;
        a.epsilon_exploration = args.uct_epsilon;
        a.seed = seed;
        auto mgr = std::make_shared<mcts::UctManager>(a);
        auto root = std::make_shared<mcts::MaxUctDNode>(mgr, init_state, 0, 0);
        return {std::static_pointer_cast<mcts::MctsManager>(mgr),
                std::static_pointer_cast<mcts::MctsDNode>(root)};
    }
    if (args.algo == "PowerUCT") {
        mcts::PowerUctManagerArgs a(env, args.power_mean);
        a.max_depth = max_depth;
        a.mcts_mode = false;
        a.bias = (args.uct_bias < 0.0) ? mcts::UctManagerArgs::USE_AUTO_BIAS : args.uct_bias;
        a.recommend_most_visited = false;
        a.epsilon_exploration = args.uct_epsilon;
        a.seed = seed;
        auto mgr = std::make_shared<mcts::PowerUctManager>(a);
        auto root = std::make_shared<mcts::PowerUctDNode>(mgr, init_state, 0, 0);
        return {std::static_pointer_cast<mcts::MctsManager>(mgr),
                std::static_pointer_cast<mcts::MctsDNode>(root)};
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
        auto mgr = std::make_shared<mcts::CatsoManager>(a);
        auto root = std::make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
        return {std::static_pointer_cast<mcts::MctsManager>(mgr),
                std::static_pointer_cast<mcts::MctsDNode>(root)};
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
    auto mgr = std::make_shared<mcts::PatsoManager>(a);
    auto root = std::make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
    return {std::static_pointer_cast<mcts::MctsManager>(mgr),
            std::static_pointer_cast<mcts::MctsDNode>(root)};
}

// -------------------------------------------------------- evaluate_tree (MC)
//
// CatastropheFn may be either:
//   bool fn(std::shared_ptr<const mcts::State>)
//   bool fn(std::shared_ptr<const mcts::State>,
//           std::shared_ptr<const mcts::Action>,
//           std::shared_ptr<const mcts::State>)
// Each env binary supplies the form needed for its catastrophic-count metric.

template <typename CatastropheFn>
requires requires (const CatastropheFn& fn, std::shared_ptr<const mcts::State> s) {
    fn(s);
}
bool catastrophe_from_initial_state(
    const CatastropheFn& catastrophe_fn,
    std::shared_ptr<const mcts::State> state)
{
    return catastrophe_fn(state);
}

template <typename CatastropheFn>
requires (!requires (const CatastropheFn& fn, std::shared_ptr<const mcts::State> s) {
    fn(s);
})
bool catastrophe_from_initial_state(
    const CatastropheFn& catastrophe_fn,
    std::shared_ptr<const mcts::State> state)
{
    (void)catastrophe_fn;
    (void)state;
    return false;
}

template <typename CatastropheFn>
requires requires (
    const CatastropheFn& fn,
    std::shared_ptr<const mcts::State> s,
    std::shared_ptr<const mcts::Action> a,
    std::shared_ptr<const mcts::State> ns) {
    fn(s, a, ns);
}
bool catastrophe_from_transition(
    const CatastropheFn& catastrophe_fn,
    std::shared_ptr<const mcts::State> state,
    std::shared_ptr<const mcts::Action> action,
    std::shared_ptr<const mcts::State> next_state)
{
    return catastrophe_fn(state, action, next_state);
}

template <typename CatastropheFn>
requires (!requires (
    const CatastropheFn& fn,
    std::shared_ptr<const mcts::State> s,
    std::shared_ptr<const mcts::Action> a,
    std::shared_ptr<const mcts::State> ns) {
    fn(s, a, ns);
})
bool catastrophe_from_transition(
    const CatastropheFn& catastrophe_fn,
    std::shared_ptr<const mcts::State> state,
    std::shared_ptr<const mcts::Action> action,
    std::shared_ptr<const mcts::State> next_state)
{
    (void)state;
    (void)action;
    return catastrophe_fn(next_state);
}

template <typename EnvT, typename CatastropheFn>
PerSeedMetrics evaluate_tree(
    std::shared_ptr<const EnvT> env,
    std::shared_ptr<const mcts::MctsDNode> root,
    int horizon,
    int eval_rollouts,
    int seed,
    double cvar_tau,
    CatastropheFn catastrophe_fn,
    bool debug_trajectories = false,
    const std::string& debug_trajectory_label = "")
{
    PerSeedMetrics m;
    mcts::RandManager eval_rng(seed);
    mcts::EvalPolicy policy(root, env, eval_rng);
    std::vector<double> sampled_returns;
    sampled_returns.reserve(static_cast<size_t>(eval_rollouts));
    int catastrophic_count = 0;
    const bool should_debug_trajectories =
        debug_trajectories || mcts::should_debug_mc_eval_trajectories_from_env();
    const std::string trajectory_label = debug_trajectory_label.empty() ? "MC" : debug_trajectory_label;

    for (int rollout = 0; rollout < eval_rollouts; ++rollout) {
        policy.reset();
        int steps = 0;
        double sample_return = 0.0;
        std::shared_ptr<const mcts::State> state = env->get_initial_state_itfc();
        bool was_catastrophic = catastrophe_from_initial_state(catastrophe_fn, state);
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
            if (catastrophe_from_transition(catastrophe_fn, state, action, next_state)) {
                was_catastrophic = true;
            }
            policy.update_step(action, observation);
            state = next_state;
            ++steps;
        }
        trajectory_trace.print(std::cerr);
        sampled_returns.push_back(sample_return);
        if (was_catastrophic) ++catastrophic_count;
    }

    if (sampled_returns.empty()) return m;
    double sum = 0.0;
    for (double r : sampled_returns) sum += r;
    m.mc_mean = sum / static_cast<double>(sampled_returns.size());
    if (sampled_returns.size() > 1u) {
        double var = 0.0;
        for (double r : sampled_returns) var += std::pow(r - m.mc_mean, 2.0);
        var /= static_cast<double>(sampled_returns.size() - 1u);
        m.mc_stddev = std::sqrt(var);
    }
    m.mc_cvar = compute_empirical_lower_tail_cvar(sampled_returns, cvar_tau);
    m.catastrophic_count = catastrophic_count;
    return m;
}

// ---------------------------------------------- root recommendation evaluator
//
// SolutionT must expose:
//   .optimal_cvar : double
//   .action_cvars : map-like<int, double>     (find(rid) -> iterator)

template <typename EnvT, typename SolutionT>
void evaluate_root_recommendation(
    std::shared_ptr<const EnvT> env,
    std::shared_ptr<const mcts::MctsDNode> root,
    const SolutionT& root_solution,
    PerSeedMetrics& m)
{
    auto context = env->sample_context_itfc(env->get_initial_state_itfc());
    auto recommended = root->recommend_action_itfc(*context);
    const int rid = std::static_pointer_cast<const mcts::IntAction>(recommended)->action;
    const auto it = root_solution.action_cvars.find(rid);
    if (it != root_solution.action_cvars.end()) {
        m.cvar_regret = std::abs(root_solution.optimal_cvar - it->second);
        if (m.cvar_regret <= kCvarTolerance) m.optimal_action_hit = 1.0;
    }
}

// --------------------------------------------------- JSON output (hand-built)

inline std::string json_double(double v) {
    if (std::isnan(v)) return "null";
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    return oss.str();
}

inline void emit_json(std::ostream& out, const AggregatedMetrics& agg) {
    out << "{"
        << "\"cvar_regret\":" << json_double(agg.mean_regret())
        << ",\"cvar_regret_stddev\":" << json_double(agg.regret_stddev())
        << ",\"mc_mean\":" << json_double(agg.mean(agg.sum_mc_mean))
        << ",\"mc_cvar\":" << json_double(agg.mean(agg.sum_mc_cvar))
        << ",\"mc_stddev\":" << json_double(agg.mean(agg.sum_mc_stddev))
        << ",\"optimal_action_prob\":" << json_double(agg.mean(agg.sum_opt_hit))
        << ",\"catastrophic_count\":" << json_double(agg.mean(agg.sum_catastrophic))
        << ",\"num_seeds\":" << agg.n
        << ",\"wall_time_sec\":" << json_double(agg.sum_wall_time)
        << "}" << std::endl;
}

}  // namespace mcts::exp::eval
