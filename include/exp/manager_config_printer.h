#pragma once

#include "algorithms/catso/catso_manager.h"
#include "algorithms/catso/patso_manager.h"
#include "algorithms/ments/dents/dents_manager.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/uct/power_uct_manager.h"
#include "algorithms/uct/uct_manager.h"
#include "algorithms/varde/varde_manager.h"
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

    inline std::string format_manager_bool(bool value) {
        return value ? "true" : "false";
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

        if (auto dents = std::dynamic_pointer_cast<const mcts::DentsManager>(mgr)) {
            std::ostringstream ss;
            ss << "temp=" << format_manager_double(dents->temp)
               << ",epsilon=" << format_manager_double(dents->epsilon)
               << ",use_dp_value=" << format_manager_bool(dents->use_dp_value);
            return ss.str();
        }

        if (auto ments = std::dynamic_pointer_cast<const mcts::MentsManager>(mgr)) {
            std::ostringstream ss;
            ss << "temp=" << format_manager_double(ments->temp)
               << ",epsilon=" << format_manager_double(ments->epsilon);
            return ss.str();
        }

        if (auto rvip = std::dynamic_pointer_cast<const mcts::RvipManager>(mgr)) {
            std::ostringstream ss;
            ss << "temp=" << format_manager_double(rvip->temp)
               << ",variance_floor=" << format_manager_double(rvip->variance_floor)
               << ",use_dp_value=" << format_manager_bool(rvip->use_dp_value);
            return ss.str();
        }

        return "";
    }
}
