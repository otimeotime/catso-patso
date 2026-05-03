// Single-config evaluator for Guarded Maze, driven by tune/tune.py.
//
// Uses the env-agnostic helpers in include/exp/eval_common.h plus the template
// CvarOracle from include/exp/discrete_runner_common.h. Catastrophe predicate
// mirrors run_guarded_maze.cpp's lambda (last_guard_outcome ∈ {0, 1}, i.e. the
// -60 and -30 tail outcomes).

#include "env/guarded_maze_env.h"
#include "exp/discrete_runner_common.h"
#include "exp/eval_common.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <tuple>

using namespace std;
using namespace mcts::exp::eval;

int main(int argc, char** argv) {
    // Default horizon matches run_guarded_maze.cpp (max_steps=500).
    const CliArgs args = parse_args(argc, argv, /*default_horizon=*/500);

    auto env = make_shared<mcts::exp::GuardedMazeEnv>();

    cerr << "[eval-guarded-maze] algo=" << args.algo
         << ", cvar_tau=" << args.cvar_tau
         << ", trial_budget=" << args.trial_budget
         << ", num_seeds=" << args.num_seeds
         << ", eval_rollouts=" << args.eval_rollouts
         << ", threads=" << args.threads
         << ", horizon=" << args.horizon
         << "\n";

    mcts::exp::runner::CvarOracle<mcts::exp::GuardedMazeEnv, mcts::Int4TupleState> oracle(
        static_pointer_cast<const mcts::exp::GuardedMazeEnv>(env), args.cvar_tau);
    const auto& root_solution = oracle.solve_state(env->get_initial_state());
    cerr << "[eval-guarded-maze] root_optimal_cvar=" << root_solution.optimal_cvar << "\n";

    auto init_state = env->get_initial_state_itfc();
    AggregatedMetrics agg;

    // Same catastrophe predicate as run_guarded_maze.cpp:
    //   last_guard_outcome < neutral (=2)  →  outcomes {0, 1} = -60, -30
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        const auto typed = static_pointer_cast<const mcts::Int4TupleState>(state);
        return get<3>(typed->state) < mcts::exp::GuardedMazeEnv::neutral_guard_reward_outcome;
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
            static_pointer_cast<const mcts::exp::GuardedMazeEnv>(env),
            root,
            args.horizon,
            args.eval_rollouts,
            seed + 777,
            args.cvar_tau,
            catastrophe_fn,
            args.debug_trajectories,
            args.algo);
        evaluate_root_recommendation(
            static_pointer_cast<const mcts::exp::GuardedMazeEnv>(env),
            root, root_solution, m);
        m.wall_time_sec =
            chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - t0).count();

        cerr << "[eval-guarded-maze] seed " << (s + 1) << "/" << args.num_seeds
             << " regret=" << m.cvar_regret
             << " mc_cvar=" << m.mc_cvar
             << " mc_mean=" << m.mc_mean
             << " (" << fixed << setprecision(2) << m.wall_time_sec << "s)\n";

        agg.add(m);
    }

    emit_json(cout, agg);
    return 0;
}
