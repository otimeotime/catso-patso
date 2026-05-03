// Single-config evaluator for Lunar Lander, driven by tune/tune.py.
//
// Uses the env-agnostic helpers in include/exp/eval_common.h plus the template
// CvarOracle from include/exp/discrete_runner_common.h. Compiles against
// LunarLanderState only after the .state tuple aggregator is added to that
// class (see include/env/lunar_lander_env.h).
//
// CAUTION: Lunar Lander's reachable state space is large. The exhaustive
// CvarOracle may consume substantial memory and wall-clock per invocation.
// Benchmark before launching long tuning runs.

#include "env/lunar_lander_env.h"
#include "exp/discrete_runner_common.h"
#include "exp/eval_common.h"

#include <chrono>
#include <iostream>
#include <memory>

using namespace std;
using namespace mcts::exp::eval;

int main(int argc, char** argv) {
    // Default horizon mirrors the env's default max_steps=200.
    const CliArgs args = parse_args(argc, argv, /*default_horizon=*/200);

    auto env = make_shared<mcts::exp::LunarLanderEnv>(/*max_steps=*/args.horizon);

    cerr << "[eval-lunar-lander] algo=" << args.algo
         << ", cvar_tau=" << args.cvar_tau
         << ", trial_budget=" << args.trial_budget
         << ", num_seeds=" << args.num_seeds
         << ", eval_rollouts=" << args.eval_rollouts
         << ", threads=" << args.threads
         << ", horizon=" << args.horizon
         << "\n";

    const auto t_oracle_start = chrono::steady_clock::now();
    mcts::exp::runner::CvarOracle<mcts::exp::LunarLanderEnv, mcts::exp::LunarLanderState> oracle(
        static_pointer_cast<const mcts::exp::LunarLanderEnv>(env), args.cvar_tau);
    const auto& root_solution = oracle.solve_state(env->get_initial_state());
    const double oracle_secs =
        chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - t_oracle_start).count();
    cerr << "[eval-lunar-lander] root_optimal_cvar=" << root_solution.optimal_cvar
         << " (oracle solve: " << fixed << setprecision(2) << oracle_secs << "s)\n";

    auto init_state = env->get_initial_state_itfc();
    AggregatedMetrics agg;

    const auto catastrophe_fn = [env_ptr = env](shared_ptr<const mcts::State> s) {
        return env_ptr->is_catastrophic_state_itfc(s);
    };

    for (int s = 0; s < args.num_seeds; ++s) {
        const int seed = args.base_seed
            + 1000 * s
            + static_cast<int>(hash<string>{}(args.algo) % 997);
        const auto t0 = chrono::steady_clock::now();

        auto [mgr, root] = build_planner(args, static_pointer_cast<mcts::MctsEnv>(env), init_state, seed);
        auto pool = make_unique<mcts::MctsPool>(mgr, root, args.threads);
        pool->run_trials(args.trial_budget, numeric_limits<double>::max(), true);

        PerSeedMetrics m = evaluate_tree(
            static_pointer_cast<const mcts::exp::LunarLanderEnv>(env),
            root,
            args.horizon,
            args.eval_rollouts,
            seed + 777,
            args.cvar_tau,
            catastrophe_fn,
            args.debug_trajectories,
            args.algo);
        evaluate_root_recommendation(
            static_pointer_cast<const mcts::exp::LunarLanderEnv>(env),
            root, root_solution, m);
        m.wall_time_sec =
            chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - t0).count();

        cerr << "[eval-lunar-lander] seed " << (s + 1) << "/" << args.num_seeds
             << " regret=" << m.cvar_regret
             << " mc_cvar=" << m.mc_cvar
             << " mc_mean=" << m.mc_mean
             << " (" << fixed << setprecision(2) << m.wall_time_sec << "s)\n";

        agg.add(m);
    }

    emit_json(cout, agg);
    return 0;
}
