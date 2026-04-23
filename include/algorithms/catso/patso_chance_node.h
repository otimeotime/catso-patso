#pragma once

#include "algorithms/catso/catso_chance_node.h"
#include "algorithms/catso/patso_manager.h"

#include <memory>
#include <vector>

namespace mcts {
    class PatsoDNode;

    class PatsoCNode : public CatsoCNode {
        friend PatsoDNode;

        protected:
            static constexpr double particle_match_tolerance = 1e-12;

            std::vector<double> particles;
            std::vector<double> particle_weights;

            int find_particle_index(double q_sample) const;
            void maybe_merge_particles();
            virtual void update_distribution(double q_sample) override;
            virtual double compute_mean_value() const override;
            virtual double draw_thompson_value() const override;

        public:
            PatsoCNode(
                std::shared_ptr<PatsoManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const PatsoDNode> parent=nullptr);

            virtual ~PatsoCNode() = default;
            virtual double get_cvar_value() const override;
            virtual double sample_cvar_value() const override;

        protected:
            std::shared_ptr<CatsoDNode> create_child_node_helper(std::shared_ptr<const State> observation) const;
            virtual std::string get_pretty_print_val() const override;

        public:
            std::shared_ptr<PatsoDNode> create_child_node(std::shared_ptr<const State> observation);
            bool has_child_node(std::shared_ptr<const State> observation) const;
            std::shared_ptr<PatsoDNode> get_child_node(std::shared_ptr<const State> observation) const;
            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation,
                std::shared_ptr<const State> next_state=nullptr) const override;
    };
}
