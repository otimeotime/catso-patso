#include "env/sailing_env.h"

#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/est/est_decision_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/ments/dents/dents_manager.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"
#include "algorithms/varde/varde_decision_node.h"
#include "algorithms/varde/varde_manager.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env_context.h"
#include "exp/manager_config_printer.h"

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace {
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
    };

    static EvalStats evaluate_tree(
        shared_ptr<const mcts::MctsEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        int horizon,
        int rollouts,
        int threads,
        int seed)
    {
        mcts::RandManager eval_rng(seed);
        mcts::EvalPolicy policy(root, env, eval_rng);
        mcts::MCEvaluator evaluator(env, policy, horizon, eval_rng);
        evaluator.run_rollouts(rollouts, threads);
        return {evaluator.get_mean_return(), evaluator.get_stddev_return()};
    }

    static vector<Candidate> build_candidates() {
        vector<Candidate> cands;

        // UCT with best config: bias=10.0, eps=0.100000
        cands.push_back({"UCT", "bias=10.0,epsilon=0.100000", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::UctManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.bias = 10.0;
            args.epsilon_exploration = 0.100000;
            args.seed = seed;
            auto mgr = make_shared<mcts::UctManager>(args);
            auto root = make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"CATSO", "n_atoms=51,optimism=1.0,p=2.0,tau=0.1", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::CatsoManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.n_atoms = 51;
            args.optimism_constant = 1.0;
            args.power_mean_exponent = 2.0;
            args.cvar_tau = 0.1;
            args.seed = seed;
            auto mgr = make_shared<mcts::CatsoManager>(args);
            auto root = make_shared<mcts::CatsoDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"PATSO", "max_particles=64,optimism=1.0,p=2.0,tau=0.1", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::PatsoManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.max_particles = 64;
            args.optimism_constant = 1.0;
            args.power_mean_exponent = 2.0;
            args.cvar_tau = 0.1;
            args.seed = seed;
            auto mgr = make_shared<mcts::PatsoManager>(args);
            auto root = make_shared<mcts::PatsoDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        // MENTS with best config: temp=0.100000, eps=0.250000
        cands.push_back({"MENTS", "temp=0.100000,epsilon=0.250000", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::MentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 0.100000;
            args.epsilon = 0.250000;
            args.seed = seed;
            auto mgr = make_shared<mcts::MentsManager>(args);
            auto root = make_shared<mcts::MentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        // RENTS with best config: temp=0.100000, eps=0.250000
        cands.push_back({"RENTS", "temp=0.100000,epsilon=0.250000", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::MentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 0.100000;
            args.epsilon = 0.250000;
            args.seed = seed;
            auto mgr = make_shared<mcts::MentsManager>(args);
            auto root = make_shared<mcts::RentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        // TENTS with best config: temp=0.100000, eps=0.250000
        cands.push_back({"TENTS", "temp=0.100000,epsilon=0.250000", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::MentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 0.100000;
            args.epsilon = 0.250000;
            args.seed = seed;
            auto mgr = make_shared<mcts::MentsManager>(args);
            auto root = make_shared<mcts::TentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        // DENTS with best config: temp=2.000000, eps=0.750000
        cands.push_back({"DENTS", "temp=2.000000,epsilon=0.750000,use_dp_value=true", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::DentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 2.000000;
            args.epsilon = 0.750000;
            args.seed = seed;
            args.use_dp_value = true;
            auto mgr = make_shared<mcts::DentsManager>(args);
            auto root = make_shared<mcts::DentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        // BTS with best config: temp=3.000000, eps=0.250000
        cands.push_back({"BTS", "temp=3.000000,epsilon=0.250000", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::DentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 3.000000;
            args.epsilon = 0.250000;
            args.seed = seed;
            auto mgr = make_shared<mcts::DentsManager>(args);
            auto root = make_shared<mcts::EstDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        // VarDE with best config: temp=1.000000, variance_floor=100.000000, value_src=mean
        cands.push_back({"VarDE", "temp=1.500000,variance_floor=100.000000,use_dp_value=false", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::RvipManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 1.500000;
            args.variance_floor = 100.000000;
            args.seed = seed;
            args.use_dp_value = false;
            auto mgr = make_shared<mcts::RvipManager>(args);
            auto root = make_shared<mcts::RvipDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        return cands;
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    const int grid_size = 6;
    const int init_wind_dir = 7;
    const double wind_stay_prob = 0.6;
    const double wind_turn_prob = 0.2;
    const double base_cost = 0.1;
    const double angle_cost_scale = 1.0;
    const double goal_reward = 0.0;
    const int horizon = 50;
    const int eval_rollouts = 200;
    const int runs = 100;
    const int threads = 4;
    const int base_seed = 4242;

    // Trial counts to evaluate at
    vector<int> trial_counts;
    for (int i = 1; i <= 30; ++i) {
        trial_counts.push_back(i * 1000);
    }

    auto env = make_shared<mcts::exp::SailingEnv>(
        grid_size, init_wind_dir, wind_stay_prob, wind_turn_prob, base_cost, angle_cost_scale, goal_reward);
    auto init_state = env->get_initial_state_itfc();
    auto candidates = build_candidates();

    ofstream out("results_sailing.csv", ios::out | ios::trunc);
    out << "env,algorithm,run,trial,mc_mean,mc_stddev\n";

    cout << "[exp] Sailing " << grid_size << "x" << grid_size << ", horizon=" << horizon << "\n";
    cout << "  Algorithms: " << candidates.size() << ", Runs: " << runs << ", Trial counts: " << trial_counts.size() << "\n";

    for (const auto& cand : candidates) {
        bool printed_config = false;

        for (int run = 0; run < runs; ++run) {
            int seed = base_seed + 1000 * run + (int)hash<string>{}(cand.algo) % 997;

            // Create a fresh tree for this run
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

            // Run trials and evaluate at each checkpoint
            int trials_so_far = 0;
            for (int target_trials : trial_counts) {
                int trials_to_add = target_trials - trials_so_far;
                if (trials_to_add > 0) {
                    pool->run_trials(trials_to_add, numeric_limits<double>::max(), true);
                    trials_so_far = target_trials;
                }

                auto stats = evaluate_tree(env, root, horizon, eval_rollouts, threads, seed + 777);
                out << "sailing," << cand.algo << "," << run
                    << "," << target_trials << "," << stats.mean << "," << stats.stddev << "\n";
            }

            cout << "  " << cand.algo << " run " << (run + 1) << "/" << runs << " done\n";
        }
    }

    cout << "\nResults written to results_sailing.csv\n";
    return 0;
}
