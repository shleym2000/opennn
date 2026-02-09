//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "response_optimization.h"
#include "scaling_layer.h"
#include "bounding_layer.h"
#include "dataset.h"
#include "neural_network.h"
#include "random_utilities.h"

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network, Dataset* new_dataset)
    : dataset(new_dataset)
{
    set(new_neural_network, new_dataset);
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{   
    neural_network = new_neural_network;

    dataset = new_dataset;

    if(!neural_network) return;

    const Index outputs_number = neural_network->get_outputs_number();

    const Index raw_inputs_number  = dataset->get_variables_number("Input");
    const Index raw_outputs_number = dataset->get_variables_number("Target");

    input_conditions.resize(raw_inputs_number);
    input_conditions.setConstant(Condition::None);

    output_conditions.resize(raw_outputs_number);
    output_conditions.setConstant(Condition::None);

    if(neural_network->has("Scaling2d"))
    {
        Scaling<2>* scaling_layer = static_cast<Scaling<2>*>(neural_network->get_first("Scaling2d"));

        input_minimums = scaling_layer->get_minimums();
        input_maximums = scaling_layer->get_maximums();
    }

    // @todo from unscaling layer?

    if(neural_network->has("Bounding"))
    {
        Bounding* bounding_layer = static_cast<Bounding*>(neural_network->get_first("Bounding"));

        output_minimums = bounding_layer->get_lower_bounds();
        output_maximums = bounding_layer->get_upper_bounds();
    }
    else
    {
        output_minimums.resize(outputs_number);
        output_minimums.setZero();

        output_maximums.resize(outputs_number);
        output_maximums.setConstant(type(1));
    }
}


void ResponseOptimization::set_evaluations_number(const Index new_evaluations_number)
{
    evaluations_number = new_evaluations_number;
}


void ResponseOptimization::set_iterative_max_iterations(Index max_it)
{
    iterative_max_iterations = max_it;
}


void ResponseOptimization::set_iterative_zoom_factor(type z)
{
    iterative_zoom_factor = z;
}


void ResponseOptimization::set_iterative_min_span_eps(type eps)
{
    iterative_min_span_eps = eps;
}


void ResponseOptimization::set_iterative_improvement_tolerance(type tol)
{
    iterative_improvement_tolerance = tol;
}


Index ResponseOptimization::get_iterative_max_iterations() const
{
    return iterative_max_iterations;
}


type ResponseOptimization::get_iterative_zoom_factor() const
{
    return iterative_zoom_factor;
}


type ResponseOptimization::get_iterative_min_span_eps() const
{
    return iterative_min_span_eps;
}


type ResponseOptimization::get_iterative_improvement_tolerance() const
{
    return iterative_improvement_tolerance;
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_input_conditions() const
{
    return input_conditions;
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_output_conditions() const
{
    return output_conditions;
}


Index ResponseOptimization::get_evaluations_number() const
{
    return evaluations_number;
}


Tensor1 ResponseOptimization::get_input_minimums() const
{
    return input_minimums;
}


Tensor1 ResponseOptimization::get_input_maximums() const
{
    return input_maximums;
}


Tensor1 ResponseOptimization::get_outputs_minimums() const
{
    return output_minimums;
}


Tensor1 ResponseOptimization::get_outputs_maximums() const
{
    return output_maximums;
}


void ResponseOptimization::set_input_condition(const string& name,
                                               const ResponseOptimization::Condition& condition,
                                               const Tensor1& values)
{
    const Index raw_index = dataset->get_variable_index(name);

    if(raw_index < 0)
        throw runtime_error("Unknown input variable name: " + name);

    const vector<Index> input_indices = dataset->get_variable_indices("Input");

    auto it = find(input_indices.begin(), input_indices.end(), raw_index);

    if (it == input_indices.end())
        throw runtime_error("Variable " + name + " is not set as Input.");

    const Index relative_index = static_cast<Index>(distance(input_indices.begin(), it));

    input_conditions(relative_index) = condition;

    const vector<Dataset::Variable> raw_inputs = dataset->get_variables("Input");

    const Dataset::Variable& raw_var = raw_inputs[relative_index];
    const Index raw_inputs_number = static_cast<Index>(raw_inputs.size());

    vector<Index> raw_input_categoricals;
    vector<Index> raw_input_categories_sizes;

    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    vector<Index> raw_input_feature_start(raw_inputs_number);
    vector<Index> raw_input_feature_size(raw_inputs_number);

    Index feature_index = 0;
    size_t category_position = 0;

    for(Index current_variable = 0; current_variable < raw_inputs_number; ++current_variable)
    {
        if(category_position < raw_input_categoricals.size()
        && raw_input_categoricals[category_position] == current_variable)
        {
            const Index one_hot_size = raw_input_categories_sizes[category_position];

            raw_input_feature_start[current_variable] = feature_index;
            raw_input_feature_size[current_variable]  = one_hot_size;

            feature_index += one_hot_size;
            ++category_position;
        }
        else
        {
            raw_input_feature_start[current_variable] = feature_index;
            raw_input_feature_size[current_variable]  = 1;

            ++feature_index;
        }
    }

    const Index start = raw_input_feature_start[relative_index];
    const Index size  = raw_input_feature_size[relative_index];

    if(!raw_var.is_categorical())
    {
        if(size != 1)
            throw std::runtime_error("Non-categorical input with feature_size != 1 for " + name);

        set_input_condition(start, condition, values);

        return;
    }

    if(condition == Condition::EqualTo)
    {
        if(values.size() != 1)
            throw runtime_error("EqualTo for categorical input expects 1 numeric value (category index).");

        const Index category_index = static_cast<Index>(llround(values(0)));

        if(category_index < 0 || category_index >= size)
            throw runtime_error("Category index out of range for input " + name);

        Index feature_index = 0;
        size_t category_position = 0;

        for(Index current_variable = 0; current_variable < raw_inputs_number; ++current_variable)
        {
            const bool is_categorical =
                category_position < raw_input_categoricals.size() &&
                raw_input_categoricals[category_position] == current_variable;

            const Index one_hot_size = is_categorical
                                   ? raw_input_categories_sizes[category_position++]
                                   : 1;

            raw_input_feature_start[current_variable] = feature_index;
            raw_input_feature_size[current_variable]  = one_hot_size;

            feature_index += one_hot_size;
        }

        return;
    }

    if(condition == Condition::GreaterEqualTo
    || condition == Condition::LessEqualTo
    || condition == Condition::Between)
        throw runtime_error("Inequality conditions are not supported for categorical input " + name);
}


void ResponseOptimization::set_output_condition(const string& variable_name,
                                                const ResponseOptimization::Condition& condition,
                                                const Tensor1& values)
{
    const Index raw_index = dataset->get_variable_index(variable_name);

    if(raw_index < 0)
        throw runtime_error("Unknown output variable name: " + variable_name);

    const vector<Index> target_indices = dataset->get_variable_indices("Target");

    Index relative_index = -1;

    for(size_t i = 0; i < target_indices.size(); ++i)
    {
        if(target_indices[i] == raw_index)
        {
            relative_index = static_cast<Index>(i);
            break;
        }
    }

    if(relative_index == -1)
        throw runtime_error("Variable " + variable_name + " is not set as Target.");

    output_conditions(relative_index) = condition;

    const vector<Dataset::Variable> raw_outputs = dataset->get_variables("Target");

    const Dataset::Variable& raw_var = raw_outputs[relative_index];

    const Index raw_outputs_number  = static_cast<Index>(raw_outputs.size());

    vector<Index> raw_output_categoricals;

    vector<Index> raw_output_categories_sizes;

    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    vector<Index> raw_output_feature_start(raw_outputs_number);
    vector<Index> raw_output_feature_size(raw_outputs_number);

    Index feature_index      = 0;
    size_t category_position = 0;

    for(Index current_variable = 0; current_variable < raw_outputs_number; ++current_variable)
    {
        const bool is_categorical =
            category_position < raw_output_categoricals.size() &&
            raw_output_categoricals[category_position] == current_variable;

        const Index size = is_categorical
                               ? raw_output_categories_sizes[category_position++]
                               : 1;

        raw_output_feature_start[current_variable] = feature_index;
        raw_output_feature_size[current_variable]  = size;

        feature_index += size;
    }

    const Index start = raw_output_feature_start[relative_index];

    const Index size  = raw_output_feature_size[relative_index];

    if(!raw_var.is_categorical())
    {
        if(size != 1)
            throw runtime_error("Non-categorical output with feature_size != 1 for " + variable_name);

        set_output_condition(start, condition, values);

        return;
    }

    if(condition == Condition::EqualTo)
    {
        if(values.size() != 1)
            throw runtime_error("EqualTo for categorical output expects 1 numeric value (category index).");

        const Index category_index = static_cast<Index>(llround(values(0)));

        if(category_index < 0 || category_index >= size)
            throw runtime_error("Category index out of range for output " + variable_name);

        for(Index j = 0; j < size; ++j)
        {
            Tensor<type,1> values(1);
            values(0) = (j == category_index) ? type(1) : type(0);

            set_output_condition(start + j, Condition::EqualTo, values);
        }

        return;
    }

    if(condition == Condition::GreaterEqualTo
    || condition == Condition::LessEqualTo
    || condition == Condition::Between)
        throw runtime_error("Inequality conditions are not supported for categorical output " + variable_name);
}


void ResponseOptimization::set_input_condition(const Index index,
                                               const ResponseOptimization::Condition& condition,
                                               const Tensor1& values)
{
    switch(condition)
    {       
    case Condition::EqualTo:

        if(values.size() != 1)
            throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        input_minimums[index] = values[0];
        input_maximums[index] = values[0];

        return;

    case Condition::LessEqualTo:

        if(values.size() != 1)
            throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        input_maximums[index] = values[0];

        return;

    case Condition::GreaterEqualTo:

        if(values.size() != 1)
            throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        input_minimums[index] = values[0];

        return;

    case Condition::Between:

        if(values.size() != 2)
            throw runtime_error("For Between condition, size of values must be 2.\n");

        input_minimums[index] = values[0];
        input_maximums[index] = values[1];

        return;

    case Condition::None:

    case Condition::Minimum:

    case Condition::Maximum:

    break;
    }
}


void ResponseOptimization::set_output_condition(const Index index,
                                                const ResponseOptimization::Condition& condition,
                                                const Tensor1& values)
{
    switch(condition)
    {        
    case Condition::EqualTo:

        output_minimums[index] = values[0];
        output_maximums[index] = values[0];

        return;

    case Condition::LessEqualTo:

        if(values.size() != 1)
        throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        output_maximums[index] = values[0];

        return;

    case Condition::GreaterEqualTo:

        if(values.size() != 1)
            throw runtime_error("For GreaterEqualTo condition, size of values must be 1.\n");

        output_minimums[index] = values[0];

        return;

    case Condition::Between:

        if(values.size() != 2)
            throw runtime_error("For Between condition, size of values must be 2.\n");

        output_minimums[index] = values[0];
        output_maximums[index] = values[1];

        return;

    case Condition::None:

    case Condition::Minimum:

    case Condition::Maximum:

        break;
    }
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_conditions(const vector<string>& conditions_string) const
{
    const Index conditions_number = conditions_string.size();

    Tensor<Condition, 1> conditions(conditions_number);

    for(Index i = 0; i < conditions_number; i++)
        if(conditions_string[i] == "Minimize" || conditions_string[i] == "Minimum")
            conditions[i] = Condition::Minimum;
        else if(conditions_string[i] == "Maximize" || conditions_string[i] == "Maximum")
            conditions[i] = Condition::Maximum;
        else if(conditions_string[i] == "="|| conditions_string[i] == "EqualTo")
            conditions[i] = Condition::EqualTo;
        else if(conditions_string[i] == "Between")
            conditions[i] = Condition::Between;
        else if(conditions_string[i] == ">="
             || conditions_string[i] == ">"
             || conditions_string[i] == "GreaterEqualTo"
             || conditions_string[i] == "GreaterThan")
            conditions[i] = Condition::GreaterEqualTo;
        else if(conditions_string[i] == "<="
             || conditions_string[i] == "<"
             || conditions_string[i] == "LessEqualTo"
             || conditions_string[i] == "LessThan")
            conditions[i] = Condition::LessEqualTo;
        else
            conditions[i] = Condition::None;

    return conditions;
}


Tensor2 ResponseOptimization::calculate_inputs() const
{
    const Index inputs_number = neural_network->get_inputs_number();

    Tensor2 inputs(evaluations_number, inputs_number);
    inputs.setZero();

    const Index input_variables_number = dataset->get_variables_number("Input");

    const vector<Index> input_variables_indices = dataset->get_variable_indices("Input");

    const auto& raw_vars = dataset->get_variables("Input");

    const type tiny_span = type(1e-9);

    for(Index i = 0; i < evaluations_number; i++)
    {
        Index current_feature_index  = 0;

        for(Index j = 0; j < input_variables_number; j++)
        {
            const Index used_variable_index = input_variables_indices[j];

            const Dataset::VariableType variable_type = dataset->get_variable_type(used_variable_index);

            if(variable_type == Dataset::VariableType::Numeric
            || variable_type == Dataset::VariableType::Constant)
            {
                inputs(i, current_feature_index ) = random_uniform(input_minimums[current_feature_index ], input_maximums[current_feature_index ]);
                ++current_feature_index ;
            }
            else if(variable_type == Dataset::VariableType::Binary)
            {
                const type minimum_temp  = input_minimums[current_feature_index ];
                const type maximum_temp  = input_maximums[current_feature_index ];
                const type span = maximum_temp - minimum_temp;

                const type value = (fabs(static_cast<double>(span)) < static_cast<double>(tiny_span))
                                 ? minimum_temp
                                 : type(rand() % 2);

                inputs(i, current_feature_index ) = value;

                ++current_feature_index ;
            }
            else if(variable_type == Dataset::VariableType::Categorical)
            {
                const Index categories_number = dataset->get_variables()[used_variable_index].get_categories_number();

                Index fixed_category = -1;

                vector<Index> allowed_categories;

                for(Index category_index = 0; category_index < categories_number; ++category_index)
                {
                    const Index current_category = current_feature_index  + category_index;

                    const type minimum_catategory_temp   = input_minimums[current_category];
                    const type maximum_category_temp   = input_maximums[current_category];

                    if(minimum_catategory_temp > type(0.5))
                        fixed_category = category_index;

                    if(maximum_category_temp > type(0.5))
                        allowed_categories.push_back(category_index);

                    inputs(i, current_feature_index  + category_index) = type(0);

                }

                if(fixed_category != -1)
                {
                    inputs(i, current_feature_index  + fixed_category) = type(1);
                }
                else if(!allowed_categories.empty())
                {
                    const Index random_idx = rand() % allowed_categories.size();

                    const Index selected_cat = allowed_categories[random_idx];

                    inputs(i, current_feature_index  + selected_cat) = type(1);
                }
                else
                {
                    const Index random_category = static_cast<Index>(rand() % categories_number);

                    inputs(i, current_feature_index  + random_category) = type(1);
                }

                current_feature_index += categories_number;
            }
            else
            {
                inputs(i, current_feature_index ) = random_uniform(input_minimums[current_feature_index ], input_maximums[current_feature_index ]);

                ++current_feature_index ;
            }
        }
    }

    return inputs;
}

Tensor<type,2> ResponseOptimization::calculate_envelope(const Tensor<type,2>& inputs, const Tensor<type,2>& outputs) const
{
    const Index inputs_number = neural_network->get_inputs_number();

    const Index outputs_number = neural_network->get_outputs_number();

    Tensor2 envelope = assemble_matrix_matrix(inputs, outputs);

    if(envelope.size() == 0)
        return Tensor<type,2>();

    const Index raw_inputs_number  = dataset->get_variables_number("Input");

    const Index raw_outputs_number = dataset->get_variables_number("Target");

    vector<Index> raw_input_categoricals;

    vector<Index> raw_input_categories_sizes;

    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    const bool present_input_categorical = !raw_input_categoricals.empty();

    vector<bool> is_categorical_input(inputs_number, false);

    if(present_input_categorical)
    {
        Index feature_index = 0;

        size_t category_position = 0;

        for(Index current_variable = 0; current_variable < raw_inputs_number; ++current_variable)
        {
            if(category_position < raw_input_categoricals.size()
            && raw_input_categoricals[category_position] == current_variable)
            {
                const Index one_hot_size = raw_input_categories_sizes[category_position];

                for(Index current_category = 0; current_category < one_hot_size; ++current_category)
                {
                    if(feature_index < inputs_number)
                        is_categorical_input[feature_index] = true;

                    ++feature_index;
                }

                ++category_position;
            }
            else
            {
                if(feature_index < inputs_number)
                    is_categorical_input[feature_index] = false;

                ++feature_index;
            }
        }
    }

    vector<Index> raw_output_categoricals;

    vector<Index> raw_output_categories_sizes;

    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    const bool present_output_categorical = !raw_output_categoricals.empty();

    vector<bool> is_categorical_output(outputs_number, false);

    if(present_output_categorical)
    {
        Index output_index = 0;

        size_t category_position = 0;

        for(Index current_variable = 0; current_variable < raw_outputs_number; ++current_variable)
        {
            if(category_position < raw_output_categoricals.size()
            && raw_output_categoricals[category_position] == current_variable)
            {
                const Index one_hot_size = raw_output_categories_sizes[category_position];

                for(Index current_category = 0; current_category < one_hot_size; ++current_category)
                {
                    if(output_index < outputs_number)
                        is_categorical_output[output_index] = true;

                    ++output_index;
                }

                ++category_position;
            }
            else
            {
                if(output_index < outputs_number)
                    is_categorical_output[output_index] = false;

                ++output_index;
            }
        }
    }

    vector<Constraint> constraints;

    constraints.reserve(inputs_number + outputs_number);

    const type tiny_span = type(1e-12);

    const type equality_tol = type(1e-9);

    for(Index j = 0; j < inputs_number; ++j)
    {
        const bool is_it_categorical = present_input_categorical && is_categorical_input[j];

        const type minimum = input_minimums(j);
        const type maximum = input_maximums(j);

        if(!is_it_categorical)
            constraints.push_back({ j, minimum, maximum, false });
        else
            if(fabs(static_cast<double>(maximum - minimum)) < static_cast<double>(tiny_span))
                constraints.push_back({ j, minimum, maximum, true });
    }

    for(Index j = 0; j < outputs_number; ++j)
    {
        const bool is_it_categorical = present_output_categorical && is_categorical_output[j];

        const type minimum = output_minimums(j);
        const type maximum = output_maximums(j);

        if(!is_it_categorical)
            constraints.push_back({ inputs_number + j, minimum, maximum, false });
        else
            if(fabs(static_cast<double>(maximum - minimum)) < static_cast<double>(tiny_span))
                constraints.push_back({ inputs_number + j, minimum, maximum, true });
    }

    const Index rows_number = envelope.dimension(0);

    const Index columns_number = envelope.dimension(1);

    vector<bool> rows_to_keep_mask;

    rows_to_keep_mask.resize(static_cast<size_t>(rows_number));

    Index kept_count = 0;

    for(Index i = 0; i < rows_number; ++i)
    {
        bool fits_the_constraint = true;

        for(const auto &constraint : constraints)
        {
            const type value = envelope(i, constraint.col);

            if(!constraint.is_categorical && (value < constraint.min || value > constraint.max))
            {
                fits_the_constraint = false;
                break;
            }
            else if(fabs(static_cast<double>(value - constraint.min)) > static_cast<double>(equality_tol))
            {               
                fits_the_constraint = false;
                break;
            }
        }

        rows_to_keep_mask[static_cast<size_t>(i)] = fits_the_constraint ? 1 : 0;

        if(fits_the_constraint)
            ++kept_count;
    }

    if(kept_count == 0)
        return Tensor<type,2>();

    Tensor<type,2> filtered(kept_count, columns_number);

    Index adding_row = 0;

    for(Index i = 0; i < rows_number; ++i)
    {
        if(rows_to_keep_mask[static_cast<size_t>(i)])
        {
            const Tensor<type,1> row = envelope.chip(i, 0);

            set_row(filtered, row, adding_row);

            ++adding_row;
        }
    }

    return filtered;
}


bool ResponseOptimization::dominates_row(const Tensor<type,1>& first_row_to_compare,
                                         const Tensor<type,1>& second_row_to_compare,
                                         const Tensor<type,1>& sense)
{
    for(Index j = 0; j < first_row_to_compare.size(); ++j)
    {
        const type first_row_oriented = first_row_to_compare(j) * sense(j);

        const type second_row_oriented = second_row_to_compare(j) * sense(j);

        if(first_row_oriented > second_row_oriented)
            return false;
    }

    return true;
}


void ResponseOptimization::build_objectives_from_envelope(const Tensor<type,2>& envelope,
                                                          Tensor<type,2>& objectives,
                                                          Tensor<type,1>& sense,
                                                          Tensor<Index,1>& objective_indices) const
{
    const Index inputs_number  = neural_network->get_inputs_number();

    const Index raw_inputs_number  = dataset->get_variables_number("Input");
    const Index raw_outputs_number = dataset->get_variables_number("Target");

    const vector<Index> global_input_indices = dataset->get_variable_indices("Input");
    const vector<Index> global_output_indices = dataset->get_variable_indices("Target");

    vector<Index> raw_input_categoricals;
    vector<Index> raw_input_categories_sizes;

    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    vector<Index> raw_input_feature_start(raw_inputs_number);

    // Inputs

    Index feature_index = 0;

    size_t category_position = 0;

    for(Index i = 0; i < raw_inputs_number; ++i)
    {
        raw_input_feature_start[i] = feature_index;

        if(category_position < raw_input_categoricals.size() && raw_input_categoricals[category_position] == i)
        {
            feature_index += raw_input_categories_sizes[category_position];
            ++category_position;
        }
        else
        {
            feature_index++;
        }
    }

    vector<Index> raw_output_categoricals;

    vector<Index> raw_output_categories_sizes;

    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    vector<Index> raw_output_feature_start(raw_outputs_number);

    // Outputs

    feature_index = 0;

    category_position = 0;

    for(Index i = 0; i < raw_outputs_number; ++i)
    {
        raw_output_feature_start[i] = feature_index;

        if(category_position < raw_output_categoricals.size() && raw_output_categoricals[category_position] == i)
        {
            feature_index += raw_output_categories_sizes[category_position];
            ++category_position;
        }
        else
        {
            feature_index++;
        }
    }

    Index objectives_number = 0;

    // @todo adding bool to Index. Cast?

    for(Index i = 0; i < raw_inputs_number; ++i)
        objectives_number += (input_conditions(i) == Condition::Minimum || input_conditions(i) == Condition::Maximum);

    for(Index j = 0; j < raw_outputs_number; ++j)
        objectives_number += (output_conditions(j) == Condition::Minimum || output_conditions(j) == Condition::Maximum);

    objectives.resize(envelope.dimension(0), objectives_number);

    sense.resize(objectives_number);

    objective_indices.resize(objectives_number);

    Index counter_objectives = 0;

    for(Index i = 0; i < raw_inputs_number; ++i)
    {
        if (input_conditions(i) != Condition::Minimum && input_conditions(i) != Condition::Maximum)
            continue;

        const Index column_idx = raw_input_feature_start[i];

        for(Index row = 0; row < envelope.dimension(0); ++row)
            objectives(row, counter_objectives) = envelope(row, column_idx);

        sense(counter_objectives) = (input_conditions(i) == Condition::Maximum) ? type(-1) : type(1);

        objective_indices(counter_objectives) = global_input_indices[i];

        ++counter_objectives;
    }

    for(Index j = 0; j < raw_outputs_number; ++j)
    {
        if(output_conditions(j) != Condition::Minimum && output_conditions(j) != Condition::Maximum)
            continue;

        const Index column_idx = inputs_number + raw_output_feature_start[j];

        for(Index row = 0; row < envelope.dimension(0); ++row)
            objectives(row, counter_objectives) = envelope(row, column_idx);

        sense(counter_objectives) = (output_conditions(j) == Condition::Maximum) ? type(-1) : type(1);

        objective_indices(counter_objectives) = global_output_indices[j];

        ++counter_objectives;
    }
}


ResponseOptimization::Pareto ResponseOptimization::perform_pareto_analysis(const Tensor2& objectives,
                                                                                 const Tensor1& sense,
                                                                                 const Tensor2& inputs,
                                                                                 const Tensor2& envelope) const
{

    Pareto pareto;
//    pareto.indices = indices;
//    pareto.objectives= pareto_objective_values;
//    pareto.variables = pareto_candidate_variables;
//    pareto.inputs = pareto_input_vectors;
    pareto.envelope = envelope;

    const Index rows_number = envelope.dimension(0);

    const Index objectives_number = objectives.dimension(1);

    Tensor<bool,1> is_dominated(rows_number);
    is_dominated.setZero();

    for(Index i = 0; i < rows_number; ++i)
    {
        if(is_dominated(i))
            continue;

        for(Index k = 0; k < rows_number; ++k)
        {
            if(i == k || is_dominated(k))
                continue;

            if(dominates_row(objectives.chip(k, 0), objectives.chip(i, 0), sense))
            {
                is_dominated(i)=1;
                break;
            }

            if(dominates_row(objectives.chip(i, 0), objectives.chip(k, 0), sense))
                is_dominated(k)=1;
        }
    }

    vector<Index> indexes_non_dominate_rows;
    indexes_non_dominate_rows.reserve(rows_number);

    for(Index i = 0; i < rows_number; ++i)
        if(!is_dominated(i))
            indexes_non_dominate_rows.push_back(i);

    pareto.indices.resize(indexes_non_dominate_rows.size());

    for(Index i = 0; i < pareto.indices.size(); ++i)
        pareto.indices(i) = indexes_non_dominate_rows[i];

    const Index pareto_points_number = pareto.indices.size();

    const Index envelope_variables_number = envelope.dimension(1);

    pareto.objectives.resize(pareto_points_number, objectives_number);
    pareto.variables.resize(pareto_points_number, envelope_variables_number);
    pareto.inputs.resize(pareto_points_number, inputs.dimension(1));

    for(Index current_pareto_index = 0; current_pareto_index < pareto_points_number; ++current_pareto_index)
    {
        const Index i = pareto.indices(current_pareto_index);

        pareto.objectives.chip(current_pareto_index, 0) = objectives.chip(i, 0);
        pareto.variables.chip(current_pareto_index, 0) = envelope.chip(i, 0);
        pareto.inputs.chip(current_pareto_index, 0) = inputs.chip(i, 0);
    }

    return pareto;
}


ResponseOptimization::Pareto ResponseOptimization::perform_pareto() const
{
    const Tensor2 inputs  = calculate_inputs();

    const Tensor2 outputs = neural_network->calculate_outputs<2,2>(inputs);

    const Tensor2 envelope = calculate_envelope(inputs, outputs);

    if(envelope.size() == 0)
        return Pareto{};

    Tensor<type,2> objectives;

    Tensor<type,1> sense;

    Tensor<Index,1> objectives_indices;

    build_objectives_from_envelope(envelope, objectives, sense, objectives_indices);

    if(objectives_indices.dimension(1) < 1)
        return Pareto{};

    const Index pareto_rows_dimension = envelope.dimension(0);

    const Index input_number_dimension  = neural_network->get_inputs_number();

    Tensor<type,2> inputs_filtered(pareto_rows_dimension, input_number_dimension);

    // @todo remove these loops?

    for(Index i = 0; i < pareto_rows_dimension; ++i)
        for(Index j = 0; j < input_number_dimension; ++j)
            inputs_filtered(i,j) = envelope(i,j);

    return perform_pareto_analysis(objectives, sense, inputs_filtered, envelope);
}


ResponseOptimization::SingleOrPareto ResponseOptimization::iterative_optimization(int objective_count)
{
    constexpr type epsilon = numeric_limits<type>::epsilon();

    const vector<string> input_names  = dataset->get_variable_names("Input");

    const vector<string> output_names = dataset->get_variable_names("Target");

    const Index raw_inputs_number = input_names.size();

    const Index raw_outputs_number = output_names.size();

    const Index inputs_number = neural_network->get_inputs_number();

    const Index outputs_number = neural_network->get_outputs_number();

    vector<Index> raw_input_categoricals;

    vector<Index> raw_input_categories_sizes;

    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    vector<Index> raw_output_categoricals;

    vector<Index> raw_output_categories_sizes;

    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    vector<Index> raw_input_feature_start(raw_inputs_number);
    vector<Index> raw_input_feature_size(raw_inputs_number);

    vector<Index> raw_output_feature_start(raw_outputs_number);
    vector<Index> raw_output_feature_size(raw_outputs_number);

    vector<bool> is_categorical_input(inputs_number, false);
    vector<bool> is_categorical_output(outputs_number, false);

    {
        Index feature_index = 0;
        size_t category_position = 0;

        for(Index raw = 0; raw < raw_inputs_number; ++raw)
        {
            if(category_position < raw_input_categoricals.size() && raw_input_categoricals[category_position] == raw)
            {
                const Index one_hot_size = raw_input_categories_sizes[category_position];

                raw_input_feature_start[raw] = feature_index;
                raw_input_feature_size[raw]  = one_hot_size;

                const Index end = min(feature_index + one_hot_size, inputs_number);

                for(Index i = feature_index; i < end; ++i)
                    is_categorical_input[i] = true;

                feature_index += one_hot_size;

                ++category_position;
            }
            else
            {
                raw_input_feature_start[raw] = feature_index;
                raw_input_feature_size[raw]  = 1;

                if(feature_index < inputs_number)
                    is_categorical_input[feature_index] = false;

                ++feature_index;
            }
        }
    }

    {
        Index feature_index = 0;
        size_t category_position = 0;

        for(Index raw = 0; raw < raw_outputs_number; ++raw)
        {
            raw_output_feature_start[raw] = feature_index;

            if(category_position < raw_output_categoricals.size() && raw_output_categoricals[category_position] == raw)
            {
                const Index one_hot_size = raw_output_categories_sizes[category_position];

                raw_output_feature_size[raw]  = one_hot_size;

                const Index end = min(feature_index + one_hot_size, outputs_number);

                for(Index i = feature_index; i < end; ++i)
                    is_categorical_output[i] = true;

                feature_index += one_hot_size;

                ++category_position;
            }
            else
            {
                raw_output_feature_size[raw]  = 1;

                if(feature_index < outputs_number)
                    is_categorical_output[feature_index] = false;

                ++feature_index;
            }
        }
    }

    Tensor<type,1> original_input_minimums = input_minimums;
    Tensor<type,1> original_input_maximums = input_maximums;
    Tensor<type,1> original_output_minimums = output_minimums;
    Tensor<type,1> original_output_maximums = output_maximums;

    // @todo Try to put outside the function

    BoundsGuard guard{this, original_input_minimums,  original_input_maximums, original_output_minimums, original_output_maximums};

    int objective_column = -1;
    type sign = type(1);
    int numeric_count = 0;

    for(Index raw = 0; raw < raw_inputs_number; ++raw)
    {
        const Condition input_condition = input_conditions(raw);

        if (input_condition != Condition::Minimum && input_condition != Condition::Maximum)
            continue;

        const Index start = raw_input_feature_start[raw];
        const Index size  = raw_input_feature_size[raw];

        if (size != 1)
            continue;

        objective_column = static_cast<int>(start);
        sign = (input_condition == Condition::Minimum) ? type(1) : -type(1);
        ++numeric_count;
    }

    for(Index raw = 0; raw < raw_outputs_number; ++raw)
    {
        const auto output_condition = output_conditions(raw);

        if (output_condition != Condition::Minimum && output_condition != Condition::Maximum)
            continue;

        const Index start = raw_output_feature_start[raw];
        const Index size  = raw_output_feature_size[raw];

        if (size != 1)
            continue;

        objective_column = static_cast<int>(inputs_number + start);
        sign = (output_condition == Condition::Minimum) ? type(1) : -type(1);
        ++numeric_count;
    }

    if (objective_count <= 1 && numeric_count == 1)
    {
        Tensor<type,1> global_best_row;

        type global_best_value = (sign > 0) ? numeric_limits<double>::infinity() : -numeric_limits<double>::infinity();

        bool first_run = true;

        for(Index iteration = 0; iteration < iterative_max_iterations; ++iteration)
        {
            const Tensor<type,2> inputs = calculate_inputs();
            const Tensor<type,2> outputs = neural_network->calculate_outputs<2,2>(inputs);
            const Tensor<type,2> envelope = calculate_envelope(inputs, outputs);

            if (envelope.size() == 0)
                break;

            Tensor<type,1> obj_col = envelope.chip(objective_column, 1);

            const Index current_rows = envelope.dimension(0);

            vector<Index> sorted_indices(current_rows);

            iota(sorted_indices.begin(), sorted_indices.end(), 0);

            sort(sorted_indices.begin(), sorted_indices.end(), [&](Index a, Index b)
            {
                return (obj_col(a) * sign) < (obj_col(b) * sign);
            });

            Index best_idx_local = sorted_indices[0];
            type best_val_local  = obj_col(best_idx_local);

            if (first_run || (best_val_local * sign) < (global_best_value * sign)) {
                global_best_value = best_val_local;
                global_best_row   = envelope.chip(best_idx_local, 0);
                first_run = false;
            }

            Tensor<type,1> current_input_minimums = input_minimums;
            Tensor<type,1> current_input_maximums = input_maximums;

            const Index top_k = max(Index(1), static_cast<Index>(current_rows * iterative_zoom_factor));

            bool collapsed = true;

            for(Index raw = 0; raw < raw_inputs_number; ++raw)
            {
                const Index start = raw_input_feature_start[raw];
                const Index size  = raw_input_feature_size[raw];

                if(size > 1)
                {
                    vector<bool> keep_category(size, false);

                    for(Index k = 0; k < top_k; ++k)
                    {
                        const Index row_index = sorted_indices[k];

                        for(Index cat = 0; cat < size; ++cat)
                        {
                            if(envelope(row_index, start + cat) > 0.5)
                            {
                                keep_category[cat] = true;
                                break;
                            }
                        }
                    }

                    for(Index cat = 0; cat < size; ++cat)
                    {
                        const Index abs_idx = start + cat;

                        current_input_minimums(abs_idx) = type(0);
                        current_input_maximums(abs_idx) = keep_category[cat] ? type(1) : type(0);
                    }

                    const Index allowed_count = count(keep_category.begin(), keep_category.end(), true);

                    if(allowed_count > 1)
                        collapsed = false;
                }
                else
                {
                    const type span = current_input_maximums(start) - current_input_minimums(start);

                    const type new_span = span * iterative_zoom_factor;

                    const type center = global_best_row(start);

                    type new_min = clamp(center - new_span * type(0.5),
                                         original_input_minimums(start),
                                         original_input_maximums(start));

                    type new_max = clamp(center + new_span * type(0.5),
                                         original_input_minimums(start),
                                         original_input_maximums(start));

                    if (new_max <= new_min)
                    {
                        new_min = clamp(center - epsilon, original_input_minimums(start), original_input_maximums(start));
                        new_max = clamp(center + epsilon, original_input_minimums(start), original_input_maximums(start));
                    }

                    current_input_minimums(start) = new_min;
                    current_input_maximums(start) = new_max;

                    if (new_max - new_min > iterative_min_span_eps)
                        collapsed = false;
                }
            }

            if (collapsed)
                break;

            input_minimums = current_input_minimums;
            input_maximums = current_input_maximums;
        }

        return global_best_row;
    }

    Pareto pareto = perform_pareto();

    if(pareto.variables.size() == 0)
    {
        pareto = perform_pareto();

        return pareto;
    }

    const Tensor<type,2> all_envelope = pareto.variables;

    const Index pareto_points_number = pareto.variables.dimension(0);

    for(Index current_pareto_point = 0; current_pareto_point < pareto_points_number; ++current_pareto_point)
    {
        const Tensor<type,1> center_row = pareto.variables.chip(current_pareto_point, 0);

        Tensor<type,1> current_input_minimums = original_input_minimums;
        Tensor<type,1> current_input_maximums = original_input_maximums;

        vector<vector<bool>> allowed_categories_mask(raw_inputs_number);

        for(Index j = 0; j<raw_inputs_number; ++j)
            if(raw_input_feature_size[j] > 1)
                allowed_categories_mask[j].resize(raw_input_feature_size[j], true);

        for(Index current_iteration = 0; current_iteration < iterative_max_iterations; ++current_iteration)
        {
            bool collapsed = true;

            for(Index raw = 0; raw < raw_inputs_number; ++raw)
            {
                const Index start = raw_input_feature_start[raw];
                const Index size  = raw_input_feature_size[raw];

                if(size > 1)
                {
                    for(Index cat = 0; cat < size; ++cat)
                    {
                        const Index abs_idx = start + cat;

                        const bool original_allow = (original_input_maximums(abs_idx) > 0.5);

                        const bool population_allow = allowed_categories_mask[raw][cat];

                        current_input_minimums(abs_idx) = type(0);
                        current_input_maximums(abs_idx) = (original_allow && population_allow) ? type(1) : type(0);
                    }

                    Index count = 0;

                    for(Index cat = 0; cat < size; ++cat)
                        count += (current_input_maximums(start + cat) > type(0.5));

                    if(count > 1)
                        collapsed = false;

                    continue;
                }

                const type span = current_input_maximums(start) - current_input_minimums(start);

                const type new_span = span * iterative_zoom_factor;

                const type center = center_row(start);

                type new_min = clamp(center - new_span * type(0.5), original_input_minimums(start), original_input_maximums(start));
                type new_max = clamp(center + new_span * type(0.5), original_input_minimums(start), original_input_maximums(start));

                if(new_max <= new_min)
                {
                    new_min = clamp(center - epsilon, original_input_minimums(start), original_input_maximums(start));
                    new_max = clamp(center + epsilon, original_input_minimums(start), original_input_maximums(start));
                }

                current_input_minimums(start) = new_min;
                current_input_maximums(start) = new_max;

                if(new_max - new_min > iterative_min_span_eps)
                    collapsed = false;
            }

            input_minimums = current_input_minimums;
            input_maximums = current_input_maximums;

            const Pareto local_pareto = perform_pareto();

            // @todo append_rows is not returning anything

            if(local_pareto.variables.size() != 0)
                append_rows(all_envelope, local_pareto.variables);

            if(collapsed || local_pareto.variables.size() == 0)
                break;

            for(Index j=0; j<raw_inputs_number; ++j)
                if(!allowed_categories_mask[j].empty())
                    fill(allowed_categories_mask[j].begin(), allowed_categories_mask[j].end(), false);

            for(Index r = 0; r < local_pareto.variables.dimension(0); ++r)
            {
                for(Index raw = 0; raw < raw_inputs_number; ++raw)
                {
                    const Index size = raw_input_feature_size[raw];

                    if(size <= 1)
                        continue;

                    const Index start = raw_input_feature_start[raw];

                    for(Index category = 0; category < size; ++category)
                        if(local_pareto.variables(r, start + category) > 0.5)
                            allowed_categories_mask[raw][category] = true;
                }
            }
        }
    }

    Tensor<type,2> all_objectives;
    Tensor<type,1> sense;
    Tensor<Index,1> objective_indices;

    build_objectives_from_envelope(all_envelope, all_objectives, sense, objective_indices);

    if(objective_indices.size() < 1)
        return Pareto{};

    const Index all_rows = all_envelope.dimension(0);

    Tensor<type,2> inputs_filtered(all_rows, inputs_number);

    inputs_filtered = all_envelope.slice(array_2(0,0), array_2(all_rows, inputs_number));

    const Pareto final_pareto = perform_pareto_analysis(all_objectives, sense, inputs_filtered, all_envelope);

    return final_pareto;
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
