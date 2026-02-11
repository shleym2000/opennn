//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com
#ifndef RESPONSEOPTIMIZATION_H
#define RESPONSEOPTIMIZATION_H

#pragma once

#include "tensors.h"
#include "dataset.h"
#include "statistics.h"
#include "tinyxml2.h"

namespace opennn
{

class NeuralNetwork;

class ResponseOptimization
{
public:

    enum class ConditionType {None, Between, EqualTo, LessEqualTo, GreaterEqualTo, LessThan, GreaterThan, Minimize, Maximize};

    struct Condition
    {
        ConditionType condition;
        type low_bound;
        type up_bound;

        Condition(ConditionType new_type = ConditionType::None, type low = 0.0, type up = 0.0)
            : condition(new_type), low_bound(low), up_bound(up) {}
    };

    struct Domain
    {
        Domain() = default;
        virtual ~Domain() = default;

        Domain(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives)
        {
            set(feature_dimensions, descriptives);
        }

        void set(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives);

        void bound(const vector<Index>& feature_dimensions, const vector<Condition>& conditions);

        void reshape(const type zoom_factor,
                     const Tensor1& center,
                     const Tensor2& subset_optimal_points,
                     const vector<Index>& input_feature_dimensions,
                     const vector<Dataset::VariableType>& input_variable_types);

        Tensor1 inferior_frontier;
        Tensor1 superior_frontier;
    };

    struct FeatureSpace
    {
        FeatureSpace(const Dataset& dataset, const vector<Condition>& conditions);

        vector<Index> feature_dimensions;
        vector<Dataset::VariableType> input_variable_types;
        vector<Index> input_feature_dimensions;
        vector<Index> output_feature_dimensions;

        Domain input_domain;
        Domain output_domain;

        vector<Index> raw_objectives_indices;
        vector<int> senses_of_optimization;
        Index objectives_number;
        vector<bool> objective_is_input;
        vector<Index> objective_column_indices;
        Tensor1 utopian_point;
    };

    ResponseOptimization(NeuralNetwork* = nullptr, Dataset* = nullptr);
    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void clear_conditions();
    void set_condition(const string& name, const ConditionType condition, type low = 0.0, type up = 0.0);

    void set_iterations(const int iterations);
    void set_zoom_factor(type new_zoom_factor);
    void set_evaluations_number(const int new_evaluations_number);

    Tensor2 calculate_random_inputs(const Domain& input_domain, const FeatureSpace& feature_space) const;

    pair<Tensor2, Tensor2> filter_feasible_points(const Tensor2& inputs,
                                                  const Tensor2& outputs,
                                                  const FeatureSpace& feature_space) const;

    void normalize_objectives(Tensor2& objective_matrix,
                              Tensor1& local_utopian,
                              const FeatureSpace& feature_space) const;

    pair<Tensor2, Tensor2> calculate_subset_optimal_points(const Tensor2& feasible_inputs,
                                                           const Tensor2& feasible_outputs,
                                                           const FeatureSpace& feature_space) const;

    Tensor2 assemble_results(const Tensor2& inputs, const Tensor2& outputs, const FeatureSpace& feature_space) const;

    Tensor2 perform_single_objective_optimization(const FeatureSpace& feature_space) const;

    Tensor2 perform_multiobjective_optimization(const FeatureSpace& feature_space) const;

    pair<Tensor2, ResponseOptimization::FeatureSpace> perform_response_optimization(const vector<Condition>& conditions);

private:

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    vector<Condition> conditions;

    Index evaluations_number = 1000;

    Index max_iterations = 5;

    type zoom_factor = type(0.45);
};

}
#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
