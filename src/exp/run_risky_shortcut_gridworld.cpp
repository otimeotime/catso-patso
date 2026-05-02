#include "env/risky_shortcut_gridworld_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kGridSize = 4;
    constexpr double kSlipProb = 0.40;
    constexpr double kWindProb = 0.70;
    constexpr double kStepCost = -1.0;
    constexpr double kGoalReward = 50.0;
    constexpr double kCliffPenalty = -100.0;
    constexpr int kMaxSteps = 40;
    constexpr double kCvarTau = 0.05;
    constexpr int kEvalRollouts = 200;
    constexpr int kRuns = 3;
    constexpr int kThreads = 8;
    constexpr int kBaseSeed = 4242;
    constexpr int kCatsoAtoms = 100;
    constexpr double kCatsoOptimism = 8.0;
    constexpr double kPowerMeanExponent = 1.0;
    constexpr int kPatsoParticles = 100;
    constexpr double kPatsoOptimism = 8.0;
    constexpr double kUctEpsilon = 0.1;

    vector<int> trial_counts;
    for (int i = 1; i <= 60; ++i) {
        trial_counts.push_back(i * 1000);
    }

    auto env = make_shared<mcts::exp::RiskyShortcutGridworldEnv>(
        kGridSize,
        kSlipProb,
        kWindProb,
        vector<int>{},
        kStepCost,
        kGoalReward,
        kCliffPenalty,
        kMaxSteps);
    const string extra_info =
        "grid_size=4, max_steps=40, slip_prob=0.4, wind_prob=0.7";
    const auto catastrophe_fn = [env_ptr = env](shared_ptr<const mcts::State> state) {
        return env_ptr->is_catastrophic_state(
            static_pointer_cast<const mcts::Int3TupleState>(state));
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::RiskyShortcutGridworldEnv,
        mcts::Int3TupleState>(
        "risky_shortcut_gridworld",
        "RiskyShortcutGridworld",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_risky_shortcut_gridworld.csv",
        "results_risky_shortcut_gridworld_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kCatsoOptimism,
        kPowerMeanExponent,
        kPatsoParticles,
        kPatsoOptimism,
        kUctEpsilon);
}
