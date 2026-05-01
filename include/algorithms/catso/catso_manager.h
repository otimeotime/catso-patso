#pragma once

#include "mcts_manager.h"

namespace mcts {
    struct CatsoManagerArgs : public MctsManagerArgs {
        static const int n_atoms_default = 51;
        static constexpr double optimism_constant_default = 1.0;
        static constexpr double power_mean_exponent_default = 1.0;
        static constexpr double cvar_tau_default = 1.0;
        // Sentinel: 0.0 means "follow cvar_tau" (paper default α = τ in F9).
        static constexpr double recommend_alpha_default = 0.0;

        int n_atoms;
        double optimism_constant;
        double power_mean_exponent;
        double cvar_tau;
        double recommend_alpha;

        CatsoManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            MctsManagerArgs(mcts_env),
            n_atoms(n_atoms_default),
            optimism_constant(optimism_constant_default),
            power_mean_exponent(power_mean_exponent_default),
            cvar_tau(cvar_tau_default),
            recommend_alpha(recommend_alpha_default) {}

        virtual ~CatsoManagerArgs() = default;
    };

    class CatsoManager : public MctsManager {
        private:
            static constexpr double clamp_level(double level) {
                return level <= 0.0 ? 1e-12 : (level > 1.0 ? 1.0 : level);
            }

        public:
            int n_atoms;
            double optimism_constant;
            double power_mean_exponent;
            double cvar_tau;
            double recommend_alpha;

            CatsoManager(const CatsoManagerArgs& args) :
                MctsManager(args),
                n_atoms(args.n_atoms),
                optimism_constant(args.optimism_constant),
                power_mean_exponent(args.power_mean_exponent),
                cvar_tau(clamp_level(args.cvar_tau)),
                recommend_alpha(args.recommend_alpha <= 0.0
                    ? clamp_level(args.cvar_tau)
                    : clamp_level(args.recommend_alpha)) {}

            virtual ~CatsoManager() = default;
    };
}
