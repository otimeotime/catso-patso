#include "helper_templates.h"

#include <functional>
#include <sstream>
#include <stdexcept>

namespace mcts::helper {
    using namespace std;

    template <typename T>
    size_t hash_combine(const size_t cur_hash, const T& v) {
        return cur_hash ^ (hash<T>()(v) + 0x9e3779b9 + (cur_hash << 6) + (cur_hash >> 2));
    }

    template <typename T, typename NumericT>
    T get_max_key_break_ties_randomly(unordered_map<T,NumericT>& map, RandManager& rand_manager) {
        if (map.empty()) {
            throw runtime_error("get_max_key_break_ties_randomly: cannot select from an empty map");
        }

        auto iter = map.begin();
        NumericT best_value = iter->second;
        vector<T> best_keys{iter->first};
        ++iter;

        for (; iter != map.end(); ++iter) {
            if (iter->second > best_value) {
                best_value = iter->second;
                best_keys.clear();
                best_keys.push_back(iter->first);
            } else if (iter->second == best_value) {
                best_keys.push_back(iter->first);
            }
        }

        const int selected_index = rand_manager.get_rand_int(0, static_cast<int>(best_keys.size()));
        return best_keys[selected_index];
    }

    template <typename T>
    T sample_from_distribution(
        unordered_map<T,double>& distribution, RandManager& rand_manager, bool normalised)
    {
        if (distribution.empty()) {
            throw runtime_error("sample_from_distribution: cannot sample from an empty distribution");
        }

        double total_weight = 0.0;
        const T* fallback_key = nullptr;
        for (const pair<const T,double>& kv_pair : distribution) {
            if (kv_pair.second < 0.0) {
                throw runtime_error("sample_from_distribution: distribution contains a negative weight");
            }

            if (kv_pair.second > 0.0) {
                total_weight += kv_pair.second;
                fallback_key = &kv_pair.first;
            }
        }

        if (fallback_key == nullptr) {
            throw runtime_error("sample_from_distribution: distribution contains no positive weights");
        }

        const double draw = rand_manager.get_rand_uniform() * (normalised ? 1.0 : total_weight);
        double cumulative_weight = 0.0;
        for (const pair<const T,double>& kv_pair : distribution) {
            if (kv_pair.second <= 0.0) {
                continue;
            }

            cumulative_weight += kv_pair.second;
            if (draw < cumulative_weight) {
                return kv_pair.first;
            }
        }

        return *fallback_key;
    }

    template <typename T>
    string vector_pretty_print_string(const vector<T>& vec) {
        stringstream ss;
        ss << "[";

        bool is_first = true;
        for (const T& value : vec) {
            if (!is_first) {
                ss << ",";
            }

            ss << value;
            is_first = false;
        }

        ss << "]";
        return ss.str();
    }

    template <typename K, typename V>
    string unordered_map_pretty_print_string(const unordered_map<K,V>& mp, string delimiter) {
        stringstream ss;
        ss << "{";

        bool is_first = true;
        for (const pair<const K,V>& kv_pair : mp) {
            if (!is_first) {
                ss << ",";
            }

            ss << kv_pair.first << delimiter << kv_pair.second;
            is_first = false;
        }

        ss << "}";
        return ss.str();
    }
}
