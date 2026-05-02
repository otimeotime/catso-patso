#include "env/autonomous_vehicle_env.h"
#include "exp/discrete_runner_common.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    constexpr int kMaxSteps = 32;
    constexpr double kCvarTau = 0.05;
    constexpr int kEvalRollouts = 50;
    constexpr int kRuns = 1;
    constexpr int kThreads = 8;
    constexpr int kBaseSeed = 4242;
    constexpr int kCatsoAtoms = 100;
    constexpr double kCatsoOptimism = 2.0;
    constexpr double kPowerMeanExponent = 1.0;
    constexpr int kPatsoParticles = 128;
    constexpr double kPatsoOptimism = 4.0;
    constexpr double kUctEpsilon = 0.05;

    vector<int> trial_counts = {1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000};

    const auto horizontal_edges = mcts::exp::AutonomousVehicleEnv::default_horizontal_edges();
    const auto vertical_edges = mcts::exp::AutonomousVehicleEnv::default_vertical_edges();
    const auto edge_rewards = mcts::exp::AutonomousVehicleEnv::default_edge_rewards();
    const auto edge_probs = mcts::exp::AutonomousVehicleEnv::default_edge_probs();
    const auto reward_normalisation = mcts::exp::AutonomousVehicleEnv::default_reward_normalisation;

    auto env = make_shared<mcts::exp::AutonomousVehicleEnv>(
        horizontal_edges,
        vertical_edges,
        edge_rewards,
        edge_probs,
        kMaxSteps,
        reward_normalisation);
    const string extra_info =
        "edge_probs=[" + to_string(edge_probs[0]) + "," + to_string(edge_probs[1]) + "], max_steps=32";
    const auto catastrophe_fn = [env_ptr = env](shared_ptr<const mcts::State> state) {
        return env_ptr->is_catastrophic_state_itfc(state);
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::AutonomousVehicleEnv,
        mcts::Int3TupleState>(
        "autonomous_vehicle",
        "AutonomousVehicle",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_autonomous_vehicle.csv",
        "results_autonomous_vehicle_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kCatsoOptimism,
        kPowerMeanExponent,
        kPatsoParticles,
        kPatsoOptimism,
        kUctEpsilon);
}
