#pragma once

#include "mcts_manager.h"

namespace mcts {
    struct CatsoManagerArgs : public MctsManagerArgs {
        static const int n_atoms_default = 51;
        static constexpr double optimism_constant_default = 1.0;
        static constexpr double power_mean_exponent_default = 4.0;
        static constexpr double cvar_tau_default = 1.0;

        int n_atoms;
        double optimism_constant;
        double power_mean_exponent;
        double cvar_tau;

        CatsoManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            MctsManagerArgs(mcts_env),
            n_atoms(n_atoms_default),
            optimism_constant(optimism_constant_default),
            power_mean_exponent(power_mean_exponent_default),
            cvar_tau(cvar_tau_default) {}

        virtual ~CatsoManagerArgs() = default;
    };

    class CatsoManager : public MctsManager {
        public:
            int n_atoms;
            double optimism_constant;
            double power_mean_exponent;
            double cvar_tau;

            CatsoManager(const CatsoManagerArgs& args) :
                MctsManager(args),
                n_atoms(args.n_atoms),
                optimism_constant(args.optimism_constant),
                power_mean_exponent(args.power_mean_exponent),
                cvar_tau(args.cvar_tau <= 0.0 ? 1e-12 : (args.cvar_tau > 1.0 ? 1.0 : args.cvar_tau)) {}

            virtual ~CatsoManager() = default;
    };
}
