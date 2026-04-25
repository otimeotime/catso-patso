#include "env/river_swim_stochastic_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kMaxSteps = 200;
    constexpr double kCvarTau = 0.1;
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

    auto env = make_shared<mcts::exp::RiverSwimStochasticEnv>();
    const string extra_info = "N=7, max_steps=200, p_right_success=0.35";
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        (void)state;
        return false;
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::RiverSwimStochasticEnv,
        mcts::Int4TupleState>(
        "river_swim_stochastic",
        "RiverSwimStochastic",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_river_swim_stochastic.csv",
        "results_river_swim_stochastic_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kOptimism,
        kPowerMeanExponent,
        kPatsoParticles);
}
