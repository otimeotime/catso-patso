#include "env/overflow_queue_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kMaxSteps = 200;
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

    auto env = make_shared<mcts::exp::OverflowQueueEnv>();
    const string extra_info = "K=20, S_max=5, max_steps=200, burst_prob=0.05";
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        const auto typed_state = static_pointer_cast<const mcts::Int4TupleState>(state);
        return get<2>(typed_state->state) == mcts::exp::OverflowQueueEnv::overflow_terminal_status;
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::OverflowQueueEnv,
        mcts::Int4TupleState>(
        "overflow_queue",
        "OverflowQueue",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_overflow_queue.csv",
        "results_overflow_queue_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kOptimism,
        kPowerMeanExponent,
        kPatsoParticles);
}
