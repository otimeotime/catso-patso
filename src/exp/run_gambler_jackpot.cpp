#include "env/gambler_jackpot_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kMaxSteps = 1000;
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

    auto env = make_shared<mcts::exp::GamblerJackpotEnv>();
    const string extra_info = "capital_target=50, max_steps=1000, jackpot_prob=0.03";
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        const auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        return get<2>(typed_state->state) == mcts::exp::GamblerJackpotEnv::bankrupt_status;
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::GamblerJackpotEnv,
        mcts::Int3TupleState>(
        "gambler_jackpot",
        "GamblerJackpot",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_gambler_jackpot.csv",
        "results_gambler_jackpot_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kOptimism,
        kPowerMeanExponent,
        kPatsoParticles);
}
