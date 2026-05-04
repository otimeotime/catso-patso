#include "env/risky_ladder_env.h"
#include "exp/discrete_runner_common.h"

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    constexpr int kMaxSteps = 50;
    constexpr double kCvarTau = 0.05;
    constexpr double kDefaultDiscountGamma = 0.95;
    constexpr int kEvalRollouts = 50;
    constexpr int kRuns = 1;
    constexpr int kThreads = 8;
    constexpr int kBaseSeed = 4242;
    constexpr int kCatsoAtoms = 51;
    constexpr double kOptimism = 1.0;
    constexpr double kPowerMeanExponent = 1.0;
    constexpr int kPatsoParticles = 64;

    std::optional<double> discount_gamma_override;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gamma" && i + 1 < argc) {
            try {
                discount_gamma_override = std::stod(argv[++i]);
            }
            catch (const std::exception&) {
                cerr << "Invalid --gamma value\n";
                return 1;
            }
            if (*discount_gamma_override < 0.0 || *discount_gamma_override > 1.0) {
                cerr << "--gamma must be in [0,1]\n";
                return 1;
            }
        }
        else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [--gamma <0..1>]\n";
            return 0;
        }
        else {
            cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    const double discount_gamma = discount_gamma_override.value_or(kDefaultDiscountGamma);
    vector<int> trial_counts = {1000000};

    auto env = make_shared<mcts::exp::RiskyLadderEnv>();
    const string extra_info = "H=8, max_steps=50, fail_prob=0.15";
    const auto catastrophe_fn = [](shared_ptr<const mcts::State> state) {
        const auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        return get<2>(typed_state->state) == mcts::exp::RiskyLadderEnv::fail_status;
    };

    return mcts::exp::runner::run_experiment<
        mcts::exp::RiskyLadderEnv,
        mcts::Int3TupleState>(
        "risky_ladder",
        "RiskyLadder",
        extra_info,
        env,
        kMaxSteps,
        kCvarTau,
        kEvalRollouts,
        kRuns,
        kThreads,
        kBaseSeed,
        trial_counts,
        "results_risky_ladder.csv",
        "results_risky_ladder_summary.csv",
        catastrophe_fn,
        kCatsoAtoms,
        kOptimism,
        kPowerMeanExponent,
        kPatsoParticles,
        kOptimism,
        0.1,
        discount_gamma);
}
