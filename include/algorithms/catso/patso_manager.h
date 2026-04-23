#pragma once

#include "algorithms/catso/catso_manager.h"

namespace mcts {
    struct PatsoManagerArgs : public CatsoManagerArgs {
        static const int max_particles_default = 0;

        int max_particles;

        PatsoManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            CatsoManagerArgs(mcts_env),
            max_particles(max_particles_default) {}

        virtual ~PatsoManagerArgs() = default;
    };

    class PatsoManager : public CatsoManager {
        public:
            int max_particles;

            PatsoManager(const PatsoManagerArgs& args) :
                CatsoManager(args),
                max_particles(args.max_particles) {}

            virtual ~PatsoManager() = default;
    };
}
