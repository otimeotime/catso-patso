#pragma once

#include "mcts_manager.h"
#include "uct_manager.h"

namespace mcts {
    struct PowerUctManagerArgs : public UctManagerArgs {
        double power_mean_constant;

        PowerUctManagerArgs(std::shared_ptr<MctsEnv> mcts_env, double power_mean_constant) :
            UctManagerArgs(mcts_env),
            power_mean_constant(power_mean_constant)
        {}
    };

    class PowerUctManager: public UctManager {
        public:
            double power_mean_constant;

            PowerUctManager(const PowerUctManagerArgs& args) :
                UctManager(args),
                power_mean_constant(args.power_mean_constant)
            {}
    };
}
