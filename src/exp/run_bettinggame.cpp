#include "env/betting_game_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr double kWinProb = 0.8;
    constexpr int kMaxSequenceLength = 6;
    constexpr double kCvarTau = 0.20;
    constexpr int kEvalRollouts = 50;
    constexpr int kRuns = 3;
    constexpr int kThreads = 8;
    constexpr int kBaseSeed = 4242;
    constexpr int kCatsoAtoms = 100;
    constexpr double kCatsoOptimism = 4.0;
    constexpr double kPowerMeanExponent = 1.0;
    constexpr int kPatsoParticles = 128;
    constexpr double kPatsoOptimism = 2.0;
    constexpr double kUctEpsilon = 0.05;

    vector<int> trial_counts = {2000, 5000, 10000, 15000};

    auto env = make_shared<mcts::exp::BettingGameEnv>(
        kWinProb,
        kMaxSequenceLength,
        mcts::exp::BettingGameEnv::default_max_state_value,
        mcts::exp::BettingGameEnv::default_initial_state,
        mcts::exp::BettingGameEnv::default_reward_normalisation);
    const string extra_info =
        "win_prob=0.8, max_sequence_length=6, initial_state=16, max_state_value=256";

    return mcts::exp::runner::run_simple_experiment(
        "bettinggame",
        "BettingGame",
        extra_info,
        env,
        kMaxSequenceLength,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_bettinggame.csv",
        "results_bettinggame_summary.csv",
        kCatsoAtoms,
        kCatsoOptimism,
        kPowerMeanExponent,
        kPatsoParticles,
        kPatsoOptimism,
        kUctEpsilon);
}
