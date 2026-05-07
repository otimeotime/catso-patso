#pragma once

#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/uct/power_uct_manager.h"
#include "algorithms/uct/uct_manager.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>

namespace mcts::exp {
    inline std::string format_manager_double(double value) {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }

    inline std::string describe_manager_config(std::shared_ptr<const mcts::MctsManager> mgr) {
        if (mgr == nullptr) {
            return "";
        }

        if (auto power_uct = std::dynamic_pointer_cast<const mcts::PowerUctManager>(mgr)) {
            std::ostringstream ss;
            ss << "bias=";
            if (power_uct->bias == mcts::UctManager::USE_AUTO_BIAS) {
                ss << "auto";
            }
            else {
                ss << format_manager_double(power_uct->bias);
            }
            ss << ",epsilon=" << format_manager_double(power_uct->epsilon_exploration)
               << ",p=" << format_manager_double(power_uct->power_mean_constant);
            return ss.str();
        }

        if (auto uct = std::dynamic_pointer_cast<const mcts::UctManager>(mgr)) {
            std::ostringstream ss;
            ss << "bias=";
            if (uct->bias == mcts::UctManager::USE_AUTO_BIAS) {
                ss << "auto";
            }
            else {
                ss << format_manager_double(uct->bias);
            }
            ss << ",epsilon=" << format_manager_double(uct->epsilon_exploration);
            return ss.str();
        }

        if (auto patso = std::dynamic_pointer_cast<const mcts::PatsoManager>(mgr)) {
            std::ostringstream ss;
            ss << "max_particles=" << patso->max_particles
               << ",optimism=" << format_manager_double(patso->optimism_constant)
               << ",p=" << format_manager_double(patso->power_mean_exponent)
               << ",cvar_tau=" << format_manager_double(patso->cvar_tau)
               << ",gamma=" << format_manager_double(patso->discount_gamma);
            return ss.str();
        }

        if (auto catso = std::dynamic_pointer_cast<const mcts::CatsoManager>(mgr)) {
            std::ostringstream ss;
            ss << "n_atoms=" << catso->n_atoms
               << ",optimism=" << format_manager_double(catso->optimism_constant)
               << ",p=" << format_manager_double(catso->power_mean_exponent)
               << ",cvar_tau=" << format_manager_double(catso->cvar_tau)
               << ",gamma=" << format_manager_double(catso->discount_gamma);
            return ss.str();
        }

        return "";
    }
}
