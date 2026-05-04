// Single-config evaluator for Risky Shortcut Gridworld, driven by tune/tune.py.
//
// Uses the env-agnostic helpers in include/exp/eval_common.h plus the template
// CvarOracle from include/exp/discrete_runner_common.h. The
// `--debug-trajectories` flag mirrors eval_autonomous_vehicle.cpp and prints
// sampled state/reward traces to stderr.

#include "env/risky_shortcut_gridworld_env.h"
#include "exp/discrete_runner_common.h"
#include "exp/eval_common.h"
#include "exp/oracles/discrete_cvar_oracle_common.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>

using namespace std;
using namespace mcts::exp::eval;

namespace {
    struct EnvArgs {
        int grid_size = 4;
        double slip_prob = 0.0;
        double wind_prob = 0.02;
        double step_cost = 0.0;
        double goal_reward = 5.0;
        double cliff_penalty = -20.0;
    };

    string format_int_vector(const vector<int>& values) {
        ostringstream out;
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << values[i];
        }
        out << "]";
        return out.str();
    }

    const char* action_name(int action) {
        switch (action) {
            case mcts::exp::RiskyShortcutGridworldEnv::up_action:
                return "UP";
            case mcts::exp::RiskyShortcutGridworldEnv::right_action:
                return "RIGHT";
            case mcts::exp::RiskyShortcutGridworldEnv::down_action:
                return "DOWN";
            case mcts::exp::RiskyShortcutGridworldEnv::left_action:
                return "LEFT";
            default:
                return "UNKNOWN";
        }
    }

    void print_env_usage(std::ostream& out, const std::string& binary_name) {
        out << "Risky Shortcut Gridworld env flags:\n"
               "  " << binary_name
            << " [planner/eval flags] [--grid-size INT] [--slip-prob FLOAT] [--wind-prob FLOAT]\n"
               "  "
            << binary_name
            << " [planner/eval flags] [--step-cost FLOAT] [--goal-reward FLOAT] [--cliff-penalty FLOAT]\n";
    }

    pair<EnvArgs, vector<char*>> parse_env_args(int argc, char** argv) {
        EnvArgs env_args;
        vector<char*> filtered_argv;
        filtered_argv.reserve(static_cast<size_t>(argc));
        filtered_argv.push_back(argv[0]);

        const string binary_name = argc > 0 ? argv[0] : "mcts-eval-risky-shortcut-gridworld";
        for (int i = 1; i < argc; ++i) {
            const string flag = argv[i];
            auto consume_value = [&](auto& out_value) {
                if (i + 1 >= argc) {
                    cerr << "missing value for flag " << flag << "\n";
                    usage(cerr, binary_name);
                    print_env_usage(cerr, binary_name);
                    std::exit(2);
                }
                const string value = argv[++i];
                using ValueT = std::decay_t<decltype(out_value)>;
                bool ok = false;
                if constexpr (std::is_same_v<ValueT, int>) {
                    ok = parse_int(value, out_value);
                }
                else {
                    ok = parse_double(value, out_value);
                }
                if (!ok) {
                    cerr << "invalid value for flag " << flag << ": " << value << "\n";
                    std::exit(2);
                }
            };

            if (flag == "--grid-size") {
                consume_value(env_args.grid_size);
            }
            else if (flag == "--slip-prob") {
                consume_value(env_args.slip_prob);
            }
            else if (flag == "--wind-prob") {
                consume_value(env_args.wind_prob);
            }
            else if (flag == "--step-cost") {
                consume_value(env_args.step_cost);
            }
            else if (flag == "--goal-reward") {
                consume_value(env_args.goal_reward);
            }
            else if (flag == "--cliff-penalty") {
                consume_value(env_args.cliff_penalty);
            }
            else {
                filtered_argv.push_back(argv[i]);
            }
        }

        return {env_args, filtered_argv};
    }
}

int main(int argc, char** argv) {
    const auto [env_args, filtered_argv] = parse_env_args(argc, argv);
    const CliArgs args = parse_args(
        static_cast<int>(filtered_argv.size()),
        const_cast<char**>(filtered_argv.data()),
        /*default_horizon=*/20,
        /*default_discount_gamma=*/0.95);

    auto env = make_shared<mcts::exp::RiskyShortcutGridworldEnv>(
        env_args.grid_size,
        env_args.slip_prob,
        env_args.wind_prob,
        vector<int>{},
        env_args.step_cost,
        env_args.goal_reward,
        env_args.cliff_penalty,
        /*max_steps=*/args.horizon);
    const vector<int> windy_cols = env->get_windy_cols();

    cerr << "[eval-risky-shortcut-gridworld] algo=" << args.algo
         << ", cvar_tau=" << args.cvar_tau
         << ", gamma=" << args.discount_gamma
         << ", trial_budget=" << args.trial_budget
         << ", num_seeds=" << args.num_seeds
         << ", eval_rollouts=" << args.eval_rollouts
         << ", threads=" << args.threads
         << ", horizon=" << args.horizon
         << "\n";
    cerr << "[eval-risky-shortcut-gridworld] env"
         << ": grid_size=" << env->get_grid_size()
         << ", max_steps=" << env->get_max_steps()
         << ", slip_prob=" << env->get_slip_prob()
         << ", wind_prob=" << env->get_wind_prob()
         << ", windy_cols=" << format_int_vector(windy_cols)
         << ", step_cost=" << env->get_step_cost()
         << ", goal_reward=" << env->get_goal_reward()
         << ", cliff_penalty=" << env->get_cliff_penalty()
         << "\n";

    mcts::exp::runner::CvarOracle<mcts::exp::RiskyShortcutGridworldEnv, mcts::Int3TupleState> oracle(
        static_pointer_cast<const mcts::exp::RiskyShortcutGridworldEnv>(env),
        args.cvar_tau,
        args.discount_gamma);
    const auto& root_solution = oracle.solve_state(env->get_initial_state());
    cerr << "[eval-risky-shortcut-gridworld] root_optimal_cvar="
         << root_solution.optimal_cvar << "\n";
    for (const auto& [action_id, action_cvar] : root_solution.action_cvars) {
        cerr << "[eval-risky-shortcut-gridworld] root_action"
             << ": " << action_name(action_id)
             << " exact_cvar=" << action_cvar << "\n";
    }

    auto init_state = env->get_initial_state_itfc();
    AggregatedMetrics agg;

    const auto catastrophe_fn = [env_ptr = env](shared_ptr<const mcts::State> state) {
        return env_ptr->is_catastrophic_state(
            static_pointer_cast<const mcts::Int3TupleState>(state));
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
            static_pointer_cast<const mcts::exp::RiskyShortcutGridworldEnv>(env),
            root,
            args.horizon,
            args.eval_rollouts,
            seed + 777,
            args.cvar_tau,
            catastrophe_fn,
            args.debug_trajectories,
            args.algo);
        evaluate_root_recommendation(
            static_pointer_cast<const mcts::exp::RiskyShortcutGridworldEnv>(env),
            root, root_solution, m);
        m.wall_time_sec =
            chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - t0).count();

        auto context = env->sample_context_itfc(init_state);
        auto recommended_action = root->recommend_action_itfc(*context);
        const int recommended_action_id =
            static_pointer_cast<const mcts::IntAction>(recommended_action)->action;
        const auto exact_action_cvar_it = root_solution.action_cvars.find(recommended_action_id);
        const double exact_action_cvar =
            exact_action_cvar_it == root_solution.action_cvars.end()
            ? numeric_limits<double>::quiet_NaN()
            : exact_action_cvar_it->second;
        const double estimated_action_cvar =
            mcts::exp::oracles::estimate_recommended_action_cvar(root, recommended_action, args.cvar_tau);

        cerr << "[eval-risky-shortcut-gridworld] seed " << (s + 1) << "/" << args.num_seeds
             << " recommended_action=" << action_name(recommended_action_id)
             << " exact_action_cvar=" << exact_action_cvar
             << " estimated_action_cvar=" << estimated_action_cvar
             << " regret=" << m.cvar_regret
             << " mc_cvar=" << m.mc_cvar
             << " mc_mean=" << m.mc_mean
             << " (" << fixed << setprecision(2) << m.wall_time_sec << "s)\n";

        agg.add(m);
    }

    emit_json(cout, agg);
    return 0;
}
