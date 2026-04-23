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
    class CatsoCNode;
    class CatsoManager;

    class CatsoDNode : public MctsDNode {
        friend CatsoCNode;

        protected:
            double value_estimate;
            std::shared_ptr<ActionVector> actions;

            double compute_optimism_bonus(int parent_visits, int child_visits) const;
            double shifted_weighted_power_mean(
                const std::vector<std::pair<double,double>>& weighted_values,
                double exponent) const;
            double compute_power_mean_value() const;
            virtual void fill_ts_values(
                std::unordered_map<std::shared_ptr<const Action>,double>& action_values,
                MctsEnvContext& ctx) const;
            virtual std::shared_ptr<const Action> select_action_ts(MctsEnvContext& ctx);
            std::shared_ptr<const Action> recommend_action_best_cvar() const;

        public:
            CatsoDNode(
                std::shared_ptr<CatsoManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const CatsoCNode> parent=nullptr);

            virtual void visit(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> select_action(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> recommend_action(MctsEnvContext& ctx) const;
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

        protected:
            virtual std::shared_ptr<CatsoCNode> create_child_node_helper(
                std::shared_ptr<const Action> action) const;
            virtual std::string get_pretty_print_val() const;

        public:
            virtual ~CatsoDNode() = default;

            std::shared_ptr<CatsoCNode> create_child_node(std::shared_ptr<const Action> action);
            bool has_child_node(std::shared_ptr<const Action> action) const;
            std::shared_ptr<CatsoCNode> get_child_node(std::shared_ptr<const Action> action) const;

            virtual void visit_itfc(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> select_action_itfc(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> recommend_action_itfc(MctsEnvContext& ctx) const;
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}
