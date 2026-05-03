#pragma once

#include "exp/experiment_spec.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcts::exp {
    using EnvSpecFactory = std::function<ExperimentSpec()>;

    class EnvRegistry {
        public:
            static EnvRegistry& instance();

            void register_env(const std::string& name, EnvSpecFactory factory);
            ExperimentSpec make(const std::string& name) const;
            std::vector<std::string> names() const;

        private:
            std::unordered_map<std::string, EnvSpecFactory> factories;
    };

    void register_all_envs();
}
