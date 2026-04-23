#pragma once

#include "mcts_chance_node.h"
#include "mcts_decision_node.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcts {
    class CatsoDNode;
    class CatsoManager;

    class CatsoCNode : public MctsCNode {
        friend CatsoDNode;

        protected:
            static constexpr double initial_q_min = 0.0;
            static constexpr double initial_q_max = 0.001;
            static constexpr double match_tolerance = 1e-12;

            int num_backups;
            double mean_value;
            std::shared_ptr<StateDistr> next_state_distr;
            std::string selected_observation_key;

            int n_atoms;
            double q_min;
            double q_max;
            std::vector<double> atoms;
            std::vector<double> dirichlet_counts;

            void reset_atom_grid();
            int nearest_index(const std::vector<double>& support, double value) const;
            std::vector<double> sample_dirichlet(const std::vector<double>& alpha) const;
            double compute_lower_tail_cvar(
                const std::vector<std::pair<double,double>>& value_weight_pairs,
                double tau) const;
            virtual void update_distribution(double q_sample);
            virtual double compute_mean_value() const;
            virtual double draw_thompson_value() const;
            std::shared_ptr<const State> sample_observation_random();

        public:
            CatsoCNode(
                std::shared_ptr<CatsoManager> catso_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const CatsoDNode> parent=nullptr);

            virtual void visit(MctsEnvContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MctsEnvContext& ctx);
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);
            virtual double get_cvar_value() const;
            virtual double get_mean_value() const;
            virtual double sample_cvar_value() const;
            virtual double sample_thompson_value() const;

        protected:
            virtual std::shared_ptr<CatsoDNode> create_child_node_helper(
                std::shared_ptr<const State> observation) const;
            virtual std::string get_pretty_print_val() const;

        public:
            virtual ~CatsoCNode() = default;

            std::shared_ptr<CatsoDNode> create_child_node(std::shared_ptr<const State> observation);
            bool has_child_node(std::shared_ptr<const State> observation) const;
            std::shared_ptr<CatsoDNode> get_child_node(std::shared_ptr<const State> observation) const;

            virtual void visit_itfc(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Observation> sample_observation_itfc(MctsEnvContext& ctx);
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation,
                std::shared_ptr<const State> next_state=nullptr) const;
    };
}
