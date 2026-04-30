#include "env/guarded_maze_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kMaxSteps = 500;
    constexpr double kCvarTau = 0.05;
    constexpr int kEvalRollouts = 200;
    constexpr int kRuns = 10;
    constexpr int kThreads = 8;
    constexpr int kBaseSeed = 4242;
    constexpr int kCatsoAtoms = 51;
    constexpr double kOptimism = 1.0;
    constexpr double kPowerMeanExponent = 1.0;
    constexpr int kPatsoParticles = 64;

    vector<int> trial_counts;
    for (int i = 1; i <= 60; ++i) {
        trial_counts.push_back(i * 1000);
    }

    auto env = make_shared<mcts::exp::GuardedMazeEnv>();
    const string extra_info =
        "grid=8x8, start=(6,1), goal=(5,6), guard=(6,5), max_steps=500, deduction_limit=32";
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        const auto typed_state = static_pointer_cast<const mcts::Int4TupleState>(state);
        return get<3>(typed_state->state) < mcts::exp::GuardedMazeEnv::neutral_guard_reward_outcome;
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::GuardedMazeEnv,
        mcts::Int4TupleState>(
        "guarded_maze",
        "GuardedMaze",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_guarded_maze.csv",
        "results_guarded_maze_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kOptimism,
        kPowerMeanExponent,
        kPatsoParticles);
}
