//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RESPONSEOPTIMIZATION_H
#define RESPONSEOPTIMIZATION_H

#include <variant>
namespace opennn
{

class Dataset;
class NeuralNetwork;

struct ResponseOptimizationResults;

class ResponseOptimization
{

public:

    enum class Condition { None, Between, EqualTo, LessEqualTo, GreaterEqualTo, Minimum, Maximum };

    ResponseOptimization(NeuralNetwork* = nullptr, Dataset* = nullptr);

    Tensor<Condition, 1> get_input_conditions() const;

    Tensor<Condition, 1> get_output_conditions() const;

    Index get_evaluations_number() const;

    Tensor<type, 1> get_input_minimums() const;

    Tensor<type, 1> get_input_maximums() const;

    Tensor<type, 1> get_outputs_minimums() const;

    Tensor<type, 1> get_outputs_maximums() const;

    Tensor<Condition, 1> get_conditions(const vector<string>&) const;

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set_evaluations_number(const Index&);

    void set_input_condition(const string&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());
    void set_output_condition(const string&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());

    void set_input_condition(const Index&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());
    void set_output_condition(const Index&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());

    Tensor<type, 2> calculate_inputs() const;

    Tensor<type, 2> calculate_envelope(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

    // @todo define better, with the sizes

    struct Pareto
    {
        Pareto(const Index& points_number = 0, const Index& objectives_number = 0, const Index& envelope_variables_number = 0)
        {
            set(points_number, objectives_number, envelope_variables_number);
        }

        virtual ~Pareto() = default;

        void set(const Index& points_number = 0, const Index& objectives_number = 0, const Index& envelope_variables_number = 0)
        {
            indices.resize(points_number);
            objectives.resize(points_number, objectives_number);
            variables.resize(points_number, envelope_variables_number);
            //Tensor<type,2> pareto_inputs(points_number, inputs.dimension(1));

        }

        void print() const;

        Tensor<Index, 1> indices;
        Tensor<type, 2> objectives;
        Tensor<type, 2> variables;
        Tensor<type, 2> inputs;
        Tensor<type, 2> envelope;
    };


    struct Constraint
    {
        Index col;

        type min;

        type max;

        bool  is_categorical;
    };

    struct BoundsGuard
    {
        ResponseOptimization* self;

        Tensor<type,1> saved_input_minimums;

        Tensor<type,1> saved_input_maximums;

        Tensor<type,1> saved_output_minimums;

        Tensor<type,1> saved_output_maximums;

        // ~BoundsGuard()
        // {
        //     self->input_minimums  = saved_input_minimums;
        //     self->input_maximums  = saved_input_maximums;
        //     self->output_minimums = saved_output_minimums;
        //     self->output_maximums = saved_output_maximums;
        // }
    };

    Pareto perform_pareto() const;

    Tensor<type, 1> get_nearest_point_to_utopian(const Pareto&) const;

    using SingleOrPareto = std::variant<Tensor<type,1>, Pareto>;

    SingleOrPareto iterative_optimization(int objective_count);

    void set_iterative_max_iterations(Index);
    void set_iterative_zoom_factor(type);
    void set_iterative_min_span_eps(type);
    void set_iterative_improvement_tolerance(type);

    Index get_iterative_max_iterations() const;
    type  get_iterative_zoom_factor() const;
    type  get_iterative_min_span_eps() const;
    type  get_iterative_improvement_tolerance() const;

private:

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    Tensor<Condition, 1> input_conditions;

    Tensor<Condition, 1> output_conditions;

    Index evaluations_number = 1000;

    Tensor<type, 1> input_minimums;

    Tensor<type, 1> input_maximums;

    Tensor<type, 1> output_minimums;

    Tensor<type, 1> output_maximums;

    Index iterative_max_iterations = 12;
    type  iterative_zoom_factor = type(0.45);
    type  iterative_min_span_eps = type(1e-9);
    type  iterative_improvement_tolerance = type(1e-6);

    static bool dominates_row(const Tensor<type,1>& a,
                              const Tensor<type,1>& b,
                              const Tensor<type,1>& sense);

    Pareto perform_pareto_analysis(const Tensor<type, 2>& objectives,
                                   const Tensor<type, 1>& sense,
                                   const Tensor<type, 2>& inputs,
                                   const Tensor<type, 2>& envelope) const;

    void build_objectives_from_envelope(const Tensor<type,2>& envelope,
                                        Tensor<type,2>& objectives,
                                        Tensor<type,1>& sense,
                                        Tensor<Index,1>& objective_output_indices) const;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
