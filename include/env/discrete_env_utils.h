#pragma once

#include <memory>
#include <random>

namespace mcts::exp::detail {
    template <typename DistrT, typename KeyT>
    void accumulate_probability(const std::shared_ptr<DistrT>& distr, const KeyT& key, double prob) {
        if (prob <= 0.0) {
            return;
        }

        auto iter = distr->find(key);
        if (iter == distr->end()) {
            distr->emplace(key, prob);
        } else {
            iter->second += prob;
        }
    }

    inline std::mt19937 make_seeded_rng(int seed) {
        if (seed == 0) {
            std::random_device rd;
            return std::mt19937(rd());
        }
        return std::mt19937(seed);
    }
}
