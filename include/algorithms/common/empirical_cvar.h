#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mcts {
    inline double compute_empirical_lower_tail_cvar_from_samples(
        std::vector<double> samples,
        double alpha,
        double tolerance = 1e-12)
    {
        if (samples.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::sort(samples.begin(), samples.end());

        const double alpha_clamped = std::max(tolerance, std::min(alpha, 1.0));
        const double tail_mass = alpha_clamped * static_cast<double>(samples.size());
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;

        for (double sample : samples) {
            const double remaining_mass = std::max(0.0, tail_mass - cumulative_mass);
            const double mass_to_take = std::min(1.0, remaining_mass);
            if (mass_to_take <= 0.0) {
                break;
            }

            tail_sum += mass_to_take * sample;
            cumulative_mass += mass_to_take;

            if (cumulative_mass >= tail_mass) {
                break;
            }
        }

        if (cumulative_mass <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return tail_sum / cumulative_mass;
    }
}
