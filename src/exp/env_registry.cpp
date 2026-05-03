#include "exp/env_registry.h"

#include <algorithm>
#include <stdexcept>

namespace mcts::exp {
    EnvRegistry& EnvRegistry::instance() {
        static EnvRegistry registry;
        return registry;
    }

    void EnvRegistry::register_env(const std::string& name, EnvSpecFactory factory) {
        factories[name] = std::move(factory);
    }

    ExperimentSpec EnvRegistry::make(const std::string& name) const {
        const auto it = factories.find(name);
        if (it == factories.end()) {
            throw std::runtime_error("Unknown environment: " + name);
        }
        return it->second();
    }

    std::vector<std::string> EnvRegistry::names() const {
        std::vector<std::string> result;
        result.reserve(factories.size());
        for (const auto& [name, factory] : factories) {
            (void)factory;
            result.push_back(name);
        }
        std::sort(result.begin(), result.end());
        return result;
    }
}
