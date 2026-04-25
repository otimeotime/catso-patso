#include "env/laser_tag_safe_grid_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kMaxSteps = 400;
    constexpr double kCvarTau = 0.05;
    constexpr int kEvalRollouts = 200;
    constexpr int kRuns = 3;
    constexpr int kThreads = 8;
    constexpr int kBaseSeed = 4242;
    constexpr int kCatsoAtoms = 51;
    constexpr double kOptimism = 1.0;
    constexpr double kPowerMeanExponent = 2.0;
    constexpr int kPatsoParticles = 64;

    vector<int> trial_counts;
    for (int i = 1; i <= 20; ++i) {
        trial_counts.push_back(i * 1000);
    }

    auto env = make_shared<mcts::exp::LaserTagSafeGridEnv>();
    const string extra_info = "N=9, max_steps=400, eps_shot=0.08, slip_prob=0.1";
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        const auto typed_state = static_pointer_cast<const mcts::Int4TupleState>(state);
        return get<3>(typed_state->state) == mcts::exp::LaserTagSafeGridEnv::shot_status;
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::LaserTagSafeGridEnv,
        mcts::Int4TupleState>(
        "laser_tag_safe_grid",
        "LaserTagSafeGrid",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_laser_tag_safe_grid.csv",
        "results_laser_tag_safe_grid_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kOptimism,
        kPowerMeanExponent,
        kPatsoParticles);
}
