#include "env/two_level_risky_treasure_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kMaxSteps = 2;
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

    auto env = make_shared<mcts::exp::TwoLevelRiskyTreasureEnv>();
    const string extra_info = "stages=2, prior_success_prob=0.15, hint_error=0.2";
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        const auto typed_state = static_pointer_cast<const mcts::Int4TupleState>(state);
        return get<3>(typed_state->state) == mcts::exp::TwoLevelRiskyTreasureEnv::fail_status;
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::TwoLevelRiskyTreasureEnv,
        mcts::Int4TupleState>(
        "two_level_risky_treasure",
        "TwoLevelRiskyTreasure",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_two_level_risky_treasure.csv",
        "results_two_level_risky_treasure_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kOptimism,
        kPowerMeanExponent,
        kPatsoParticles);
}
