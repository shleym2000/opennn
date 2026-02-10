//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "pch.h"
#include "statistics.h"
#include "dataset.h"
#include "neural_network.h"
#include "response_optimization.h"
#include <algorithm>
#include <numeric>

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}

void ResponseOptimization::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    neural_network = new_neural_network;
    dataset = new_dataset;

    if(!neural_network || !dataset)
        return;

    const Index variables_number = dataset->get_variables_number();

    conditions.assign(static_cast<size_t>(variables_number), Condition(ConditionType::None));
}

void ResponseOptimization::set_condition(const string& name, const ConditionType condition, type low, type up)
{
    if(!dataset)
        throw runtime_error("Dataset not set.");

    const Index index = dataset->get_variable_index(name);

    conditions[index] = Condition(condition, low, up);
}

void ResponseOptimization::clear_conditions()
{
    if(dataset)
        conditions.assign(static_cast<size_t>(dataset->get_variables_number()), Condition(ConditionType::None));
    else
        conditions.clear();
}

void ResponseOptimization::set_evaluations_number(const int new_evaluations_number)
{
    evaluations_number = new_evaluations_number;
}
void ResponseOptimization::set_iterations(const int new_max_iterations)
{
    max_iterations = new_max_iterations;
}
void ResponseOptimization::set_zoom_factor(type new_zoom_factor)
{
    zoom_factor = new_zoom_factor;
}

void ResponseOptimization::Domain::set(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives)
{
    const Index variables_number = static_cast<Index>(feature_dimensions.size());

    const Index total_feature_dimensions = accumulate(feature_dimensions.begin(), feature_dimensions.end(), Index(0));

    inferior_frontier.resize(total_feature_dimensions);
    superior_frontier.resize(total_feature_dimensions);

    Index feature_index = 0;
    for(Index raw_variable = 0; raw_variable < variables_number; ++raw_variable)
    {
        const Index feature_dimension = feature_dimensions[raw_variable];
        if (feature_dimension > 1)
        {
            inferior_frontier.slice(array_1(feature_index), array_1(feature_dimension)).setConstant(0.0);
            superior_frontier.slice(array_1(feature_index), array_1(feature_dimension)).setConstant(1.0);
        }
        else
        {
            inferior_frontier(feature_index) = static_cast<type>(descriptives[raw_variable].minimum);
            superior_frontier(feature_index) = static_cast<type>(descriptives[raw_variable].maximum);
        }
        feature_index += feature_dimension;
    }
}

ResponseOptimization::FeatureSpace::FeatureSpace(const Dataset& dataset, const vector<Condition>& conditions)
{
    feature_dimensions = dataset.get_feature_dimensions();

    const vector<Index> input_indices = dataset.get_variable_indices("Input");
    const vector<Index> output_indices = dataset.get_variable_indices("Target");

    input_variable_types = dataset.get_variable_types(input_indices);
    input_feature_dimensions = gather_by_index(feature_dimensions, input_indices);
    output_feature_dimensions = gather_by_index(feature_dimensions, output_indices);

    const vector<Descriptives> input_descriptives = dataset.calculate_feature_descriptives("Input");
    const vector<Descriptives> output_descriptives = dataset.calculate_feature_descriptives("Target");

    const vector<Condition> input_conditions = gather_by_index(conditions, input_indices);
    const vector<Condition> output_conditions = gather_by_index(conditions, output_indices);

    input_domain.set(input_feature_dimensions, input_descriptives);
    output_domain.set(output_feature_dimensions, output_descriptives);

    input_domain.bound(input_feature_dimensions, input_conditions);
    output_domain.bound(output_feature_dimensions, output_conditions);

    objectives_number = 0;
    utopian_point.resize(input_indices.size() + output_indices.size());

    Index current_feature_index = 0;

    for (Index i = 0; i < (Index)input_indices.size(); i++)
    {
        if (input_conditions[i].condition == ConditionType::Maximize || input_conditions[i].condition == ConditionType::Minimize)
        {
            objective_is_input.push_back(true);
            objective_column_indices.push_back(current_feature_index);

            if (input_conditions[i].condition == ConditionType::Maximize)
            {
                senses_of_optimization.push_back(1);
                utopian_point(objectives_number) = input_domain.superior_frontier(current_feature_index);
            }
            else
            {
                senses_of_optimization.push_back(-1);
                utopian_point(objectives_number) = input_domain.inferior_frontier(current_feature_index);
            }
            objectives_number++;
        }
        current_feature_index += input_feature_dimensions[i];
    }

    current_feature_index = 0;

    for (Index i = 0; i < (Index)output_indices.size(); i++)
    {
        if (output_conditions[i].condition == ConditionType::Maximize || output_conditions[i].condition == ConditionType::Minimize)
        {
            objective_is_input.push_back(false);
            objective_column_indices.push_back(current_feature_index);

            if (output_conditions[i].condition == ConditionType::Maximize)
            {
                senses_of_optimization.push_back(1);
                utopian_point(objectives_number) = output_domain.superior_frontier(current_feature_index);
            }
            else
            {
                senses_of_optimization.push_back(-1);
                utopian_point(objectives_number) = output_domain.inferior_frontier(current_feature_index);
            }
            objectives_number++;
        }
        current_feature_index += output_feature_dimensions[i];
    }
    utopian_point.resize(objectives_number);
}

void ResponseOptimization::Domain::bound(const vector<Index>& feature_dimensions, const vector<Condition>& conditions)
{
    Index feature_index = 0;

    for(size_t raw_variable_index = 0; raw_variable_index < feature_dimensions.size(); ++raw_variable_index)
    {
        const Index feature_dimension = feature_dimensions[raw_variable_index];
        const Condition& condition_on_raw_variable = conditions[raw_variable_index];

        if(feature_dimension == 1)
        {
            type& inferior = inferior_frontier(feature_index);
            type& superior = superior_frontier(feature_index);

            switch(condition_on_raw_variable.condition)
            {
            case ConditionType::EqualTo:
                inferior = max(inferior, condition_on_raw_variable.low_bound);
                superior = min(superior, condition_on_raw_variable.low_bound);
                break;
            case ConditionType::Between:
                inferior = max(inferior, condition_on_raw_variable.low_bound);
                superior = min(superior, condition_on_raw_variable.up_bound);
                break;
            case ConditionType::GreaterEqualTo:
                inferior = max(inferior, condition_on_raw_variable.low_bound);
                break;
            case ConditionType::LessEqualTo:
                superior = min(superior, condition_on_raw_variable.up_bound);
                break;

            default:
                break;
            }
        }
        else if(condition_on_raw_variable.condition == ConditionType::EqualTo)
        {
            const Index category_index = static_cast<Index>(llround(condition_on_raw_variable.low_bound));

            for(Index j = 0; j < feature_dimension; ++j)
            {
                inferior_frontier(feature_index + j) = (j == category_index) ? 1.0 : 0.0;
                superior_frontier(feature_index + j) = (j == category_index) ? 1.0 : 0.0;
            }
        }
        feature_index += feature_dimension;
    }
}

Tensor2 ResponseOptimization::calculate_random_inputs(const Domain& input_domain, const FeatureSpace& feature_space) const
{
    const Index inputs_features_number = input_domain.inferior_frontier.size();
    const vector<Index> input_indices = dataset->get_variable_indices("Input");
    const vector<Dataset::VariableType> input_variable_types = dataset->get_variable_types(input_indices);

    Tensor2 random_inputs(evaluations_number, inputs_features_number);
    set_random_uniform(random_inputs, 0, 1);

    Index current_feature_index = 0;

    for(size_t raw_input_variable = 0; raw_input_variable < feature_space.input_feature_dimensions.size(); ++raw_input_variable)
    {
        const Index categories_number = feature_space.input_feature_dimensions[raw_input_variable];

        if(categories_number == 1)
        {
            if(input_variable_types[raw_input_variable] == Dataset::VariableType::Binary)
                random_inputs.chip(current_feature_index, 1) = random_inputs.chip(current_feature_index, 1).round();
            else
                random_inputs.chip(current_feature_index, 1) = random_inputs.chip(current_feature_index, 1) * (input_domain.superior_frontier(current_feature_index) - input_domain.inferior_frontier(current_feature_index)) + input_domain.inferior_frontier(current_feature_index);
            current_feature_index++;
        }
        else
        {
            random_inputs.slice(array_2(0, current_feature_index), array_2(evaluations_number, categories_number)).setZero();

            vector<Index> allowed_categories;

            for(Index i = 0; i < categories_number; ++i)
                if(input_domain.superior_frontier(current_feature_index + i) > 0.5)
                    allowed_categories.push_back(i);

            for(Index row = 0; row < evaluations_number; ++row)
                random_inputs(row, current_feature_index + allowed_categories[rand() % allowed_categories.size()]) = 1.0;

            current_feature_index += categories_number;
        }
    }
    return random_inputs;
}

void ResponseOptimization::Domain::reshape(const type zoom_factor, const Tensor1& center, const Tensor2& subset_optimal_points_inputs, const vector<Index>& input_feature_dimensions, const vector<Dataset::VariableType>& input_variable_types)
{
    Tensor1 categories_to_save = subset_optimal_points_inputs.maximum(array_1(0));

    for(Index i = 0; i < categories_to_save.size(); ++i)
        if(center(i) > categories_to_save(i))
            categories_to_save(i) = center(i);

    Index current_feature_index = 0;

    for(size_t raw_input_variable = 0; raw_input_variable < input_feature_dimensions.size(); ++raw_input_variable)
    {
        const Index categories_number = input_feature_dimensions[raw_input_variable];

        if(categories_number == 1)
        {
            if(input_variable_types[raw_input_variable] == Dataset::VariableType::Binary)
            {
                inferior_frontier(current_feature_index) = max(categories_to_save(current_feature_index), inferior_frontier(current_feature_index));
                superior_frontier(current_feature_index) = min(categories_to_save(current_feature_index), superior_frontier(current_feature_index));
            }
            else
            {
                const type half_span = (superior_frontier(current_feature_index) - inferior_frontier(current_feature_index)) * zoom_factor / 2;
                inferior_frontier(current_feature_index) = max(center(current_feature_index) - half_span, inferior_frontier(current_feature_index));
                superior_frontier(current_feature_index) = min(center(current_feature_index) + half_span, superior_frontier(current_feature_index));
            }
        }
        else
        {
            for(Index category_index = 0; category_index < categories_number; ++category_index)
            {
                const Index current_category = current_feature_index + category_index;
                inferior_frontier(current_category) = max(categories_to_save(current_category), inferior_frontier(current_category));
                superior_frontier(current_category) = min(categories_to_save(current_category), superior_frontier(current_category));
            }
        }
        current_feature_index += categories_number;
    }
}

pair<Tensor2, Tensor2> ResponseOptimization::filter_feasible_points(const Tensor2& inputs, const Tensor2& outputs, const FeatureSpace& feature_space) const
{
    const vector<Index> feasible_rows = build_feasible_rows_mask(outputs, feature_space.output_domain.inferior_frontier, feature_space.output_domain.superior_frontier);

    if(feasible_rows.empty())
        return {Tensor2(0, inputs.dimension(1)), Tensor2(0, outputs.dimension(1))};

    Tensor2 feasible_inputs((Index)feasible_rows.size(), inputs.dimension(1));
    Tensor2 feasible_outputs((Index)feasible_rows.size(), outputs.dimension(1));

    for(Index j = 0; j < (Index)feasible_rows.size(); ++j)
    {
        set_row(feasible_inputs, inputs.chip(feasible_rows[j], 0), j);
        set_row(feasible_outputs, outputs.chip(feasible_rows[j], 0), j);
    }

    return {feasible_inputs, feasible_outputs};
}

Tensor<type, 2> extract_objectives(const Tensor<type, 2>& inputs, const Tensor<type, 2>& outputs, const ResponseOptimization::FeatureSpace& feature_space)
{
    Tensor2 objective_matrix(inputs.dimension(0), feature_space.objectives_number);

    for (Index j = 0; j < feature_space.objectives_number; ++j)
        objective_matrix.chip(j, 1) = (feature_space.objective_is_input[j]) ? inputs.chip(feature_space.objective_column_indices[j], 1) * (type)feature_space.senses_of_optimization[j] : outputs.chip(feature_space.objective_column_indices[j], 1) * (type)feature_space.senses_of_optimization[j];
    return objective_matrix;
}

void ResponseOptimization::normalize_objectives(Tensor2& objective_matrix, Tensor1& local_utopian, const FeatureSpace& feature_space) const
{
    for (Index j = 0; j < feature_space.objectives_number; ++j)
    {
        Index column_index = feature_space.objective_column_indices[j];

        type inferior = feature_space.objective_is_input[j] ? feature_space.input_domain.inferior_frontier(column_index) : feature_space.output_domain.inferior_frontier(column_index);
        type superior = feature_space.objective_is_input[j] ? feature_space.input_domain.superior_frontier(column_index) : feature_space.output_domain.superior_frontier(column_index);

        type range = superior - inferior;

        if (range > 1e-12)
            objective_matrix.chip(j, 1) = objective_matrix.chip(j, 1) / range; local_utopian(j) /= range;
    }
}

pair<Tensor2, Tensor2> ResponseOptimization::calculate_subset_optimal_points(const Tensor2& feasible_inputs, const Tensor2& feasible_outputs, const FeatureSpace& feature_space) const
{
    Index subset_dimension = clamp<Index>(llround(zoom_factor * evaluations_number), 1, feasible_outputs.dimension(0));

    Tensor2 objective_matrix = extract_objectives(feasible_inputs, feasible_outputs, feature_space);
    Tensor1 local_utopian = feature_space.utopian_point;

    normalize_objectives(objective_matrix, local_utopian, feature_space);

    const Tensor<Index,1> nearest_rows = get_n_nearest_points(objective_matrix, local_utopian, (int)subset_dimension);

    Tensor2 nearest_subset_inputs(subset_dimension, feasible_inputs.dimension(1)), nearest_subset_outputs(subset_dimension, feasible_outputs.dimension(1));

    for(Index i = 0; i < subset_dimension; ++i)
    {
        nearest_subset_inputs.chip(i, 0) = feasible_inputs.chip(nearest_rows(i), 0);
        nearest_subset_outputs.chip(i, 0) = feasible_outputs.chip(nearest_rows(i), 0);
    }

    return {nearest_subset_inputs, nearest_subset_outputs};
}

Tensor2 ResponseOptimization::assemble_results(const Tensor2& inputs, const Tensor2& outputs, const FeatureSpace& feature_space) const
{
    const Index total_variables_number = (Index)feature_space.feature_dimensions.size();

    vector<Index> global_starts_blocks(total_variables_number, 0);

    for (Index i = 1; i < total_variables_number; ++i)
        global_starts_blocks[i] = global_starts_blocks[i - 1] + feature_space.feature_dimensions[i - 1];

    Tensor2 result(inputs.dimension(0), global_starts_blocks.back() + feature_space.feature_dimensions.back());

    auto copy_blocks = [&](const vector<Index>& indices_in_out, const vector<Index>& hot_encoded_dimensions, const Tensor2& source_to_copy)
    {
        Index start_source_feature_columns = 0;

        for (size_t i = 0; i < indices_in_out.size(); ++i)
        {
            result.slice(array_2(0, global_starts_blocks[indices_in_out[i]]), array_2(inputs.dimension(0), hot_encoded_dimensions[i])) = source_to_copy.slice(array_2(0, start_source_feature_columns), array_2(inputs.dimension(0), hot_encoded_dimensions[i]));
            start_source_feature_columns += hot_encoded_dimensions[i];
        }
    };

    copy_blocks(dataset->get_variable_indices("Input"), feature_space.input_feature_dimensions, inputs);
    copy_blocks(dataset->get_variable_indices("Target"), feature_space.output_feature_dimensions, outputs);

    return result;
}

Tensor2 ResponseOptimization::perform_single_objective_optimization(const FeatureSpace& feature_space) const
{
    Domain input_domain_to_iterate = feature_space.input_domain;

    pair<Tensor2, Tensor2> optimal_set;

    for (Index i = 0; i < max_iterations; i++)
    {
        Tensor2 random_inputs = calculate_random_inputs(input_domain_to_iterate, feature_space);

        auto [feasible_inputs, feasible_outputs] = filter_feasible_points(random_inputs, neural_network->calculate_outputs<2,2>(random_inputs), feature_space);

        if(feasible_inputs.dimension(0) == 0)
            break;

        optimal_set = calculate_subset_optimal_points(feasible_inputs, feasible_outputs, feature_space);
        input_domain_to_iterate.reshape(zoom_factor, optimal_set.first.chip(0,0), optimal_set.first, feature_space.input_feature_dimensions, feature_space.input_variable_types);
    }
    return optimal_set.first.dimension(0) == 0 ? Tensor2() : assemble_results(optimal_set.first, optimal_set.second, feature_space);
}

pair<Tensor2, Tensor2> calculate_pareto(const Tensor2& inputs, const Tensor2& outputs, const Tensor2& objective_matrix)
{
    const Index rows_number = objective_matrix.dimension(0);

    if (rows_number == 0)
        return {Tensor2(), Tensor2()};

    vector<bool> non_dominated(static_cast<size_t>(rows_number), true);

    for (Index i = 0; i < rows_number; ++i)
    {
        for (Index j = 0; j < rows_number; ++j)
        {
            if (i == j)
                continue;

            const Tensor<bool, 0> better_equal = (objective_matrix.chip(j, 0) >= objective_matrix.chip(i, 0)).all();

            const Tensor<bool, 0> strictly_better = (objective_matrix.chip(j, 0) > objective_matrix.chip(i, 0)).any();

            if (better_equal() && strictly_better())
            {
                non_dominated[i] = false;
                break;
            }
        }
    }
    vector<Index> non_dominated_indices;

    for (Index i = 0; i < rows_number; ++i)
        if (non_dominated[i])
            non_dominated_indices.push_back(i);

    Tensor2 pareto_inputs((Index)non_dominated_indices.size(), inputs.dimension(1));
    Tensor2 pareto_outputs((Index)non_dominated_indices.size(), outputs.dimension(1));

    for (Index i = 0; i < (Index)non_dominated_indices.size(); ++i)
    {
        pareto_inputs.chip(i, 0) = inputs.chip(non_dominated_indices[i], 0);
        pareto_outputs.chip(i, 0) = outputs.chip(non_dominated_indices[i], 0);
    }

    return {pareto_inputs, pareto_outputs};
}

Tensor2 ResponseOptimization::perform_multiobjective_optimization(const FeatureSpace& feature_space) const
{
    Tensor2 first_random_inputs = calculate_random_inputs(feature_space.input_domain, feature_space);

    auto [first_feasible_inputs, first_feasible_outputs] = filter_feasible_points(first_random_inputs, neural_network->calculate_outputs<2,2>(first_random_inputs), feature_space);

    if (first_feasible_inputs.dimension(0) == 0)
        return Tensor2();

    auto [global_pareto_inputs, global_pareto_outputs] = calculate_pareto(first_feasible_inputs, first_feasible_outputs, extract_objectives(first_feasible_inputs, first_feasible_outputs, feature_space));

    vector<Domain> local_input_domains(static_cast<size_t>(global_pareto_inputs.dimension(0)), feature_space.input_domain);

    type current_zoom = zoom_factor;

    for (Index i = 0; i < max_iterations; i++)
    {
        Tensor2 union_inputs;
        Tensor2 union_outputs;

        for (Index j = 0; j < global_pareto_inputs.dimension(0); j++)
        {
            Tensor2 local_random_inputs = calculate_random_inputs(local_input_domains[j], feature_space);

            auto [local_feasible_inputs, local_feasible_outputs] = filter_feasible_points(local_random_inputs, neural_network->calculate_outputs<2,2>(local_random_inputs), feature_space);
            auto [local_pareto_input, local_pareto_output] = calculate_pareto(local_feasible_inputs, local_feasible_outputs, extract_objectives(local_feasible_inputs, local_feasible_outputs, feature_space));

            union_inputs = append_rows(union_inputs, local_pareto_input);
            union_outputs = append_rows(union_outputs, local_pareto_output);
        }

        Tensor2 candidate_inputs = append_rows(global_pareto_inputs, union_inputs);
        Tensor2 candidate_outputs = append_rows(global_pareto_outputs, union_outputs);

        if (candidate_inputs.dimension(0) == 0)
            break;

        auto optimal_set = calculate_subset_optimal_points(candidate_inputs, candidate_outputs, feature_space);

        auto pareto_pair = calculate_pareto(candidate_inputs, candidate_outputs, extract_objectives(candidate_inputs, candidate_outputs, feature_space));

        global_pareto_inputs = pareto_pair.first; global_pareto_outputs = pareto_pair.second;

        local_input_domains.assign(static_cast<size_t>(global_pareto_inputs.dimension(0)), feature_space.input_domain);

        for (Index j = 0; j < global_pareto_inputs.dimension(0); j++)
            local_input_domains[j].reshape(current_zoom, global_pareto_inputs.chip(j, 0), optimal_set.first, feature_space.input_feature_dimensions, feature_space.input_variable_types);
        current_zoom *= 0.5;
    }
    return assemble_results(global_pareto_inputs, global_pareto_outputs, feature_space);
}

pair<Tensor2, ResponseOptimization::FeatureSpace> ResponseOptimization::perform_response_optimization(const vector<Condition>& conditions)
{
    if(!dataset)
        throw runtime_error("Dataset not set\n");

    FeatureSpace feature_space(*dataset, conditions);

    if (feature_space.objectives_number == 0)
        throw runtime_error("No objectives found\n");

    return { (feature_space.objectives_number == 1) ? perform_single_objective_optimization(feature_space) : perform_multiobjective_optimization(feature_space), feature_space };
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute iteration and/or
// modify iteration under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that iteration will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
