//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYER2D_H
#define SCALINGLAYER2D_H

#include "layer.h"
#include "statistics.h"
#include "scaling.h"

namespace opennn
{

template<int Rank> struct ScalingForwardPropagation;

template<int Rank>
class Scaling final : public Layer
{

public:

    Scaling(const dimensions& new_input_dimensions = {})
    {
        set(new_input_dimensions);
    }

    dimensions get_input_dimensions() const override
    {
        return dimensions{Index(scalers.size())};
    }

    dimensions get_output_dimensions() const override
    {
        return dimensions{Index(scalers.size())};
    }

    vector<Descriptives> get_descriptives() const
    {
        return descriptives;
    }

    Descriptives get_descriptives(const Index& index) const
    {
        return descriptives[index];
    }

    Tensor1 get_minimums() const
    {
        const Index outputs_number = get_outputs_number();

        Tensor1 minimums(outputs_number);

#pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            minimums[i] = descriptives[i].minimum;

        return minimums;
    }

    Tensor1 get_maximums() const
    {
        const Index outputs_number = get_outputs_number();

        Tensor1 maximums(outputs_number);

#pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            maximums[i] = descriptives[i].maximum;

        return maximums;
    }

    Tensor1 get_means() const
    {
        const Index outputs_number = get_outputs_number();

        Tensor1 means(outputs_number);

#pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            means[i] = descriptives[i].mean;

        return means;
    }


    Tensor1 get_standard_deviations() const
    {
        const Index outputs_number = get_outputs_number();

        Tensor1 standard_deviations(outputs_number);

        #pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            standard_deviations[i] = descriptives[i].standard_deviation;

        return standard_deviations;
    }


    vector<string> get_scalers() const
    {
        return scalers;
    }


    void set(const dimensions& new_input_dimensions = {})
    {
        if (new_input_dimensions.size() != 1)
            throw runtime_error("Input dimensions rank is not 1");

        const Index new_inputs_number = accumulate(new_input_dimensions.begin(), new_input_dimensions.end(), 1, multiplies<Index>());

        descriptives.resize(new_inputs_number);

        for(Index i = 0; i < new_inputs_number; i++)
            descriptives[i].set(type(-1.0), type(1), type(0), type(1));

        scalers.resize(new_inputs_number, "MeanStandardDeviation");

        label = "scaling_layer";

        set_scalers("MeanStandardDeviation");

        set_min_max_range(type(-1), type(1));

        name = "Scaling";

        is_trainable = false;
    }

    void set_input_dimensions(const dimensions& new_input_dimensions) override
    {
        descriptives.resize(new_input_dimensions[0]);

        scalers.resize(new_input_dimensions[0], "MeanStandardDeviation");
    }

    void set_output_dimensions(const dimensions& new_output_dimensions) override
    {
        set_input_dimensions(new_output_dimensions);
    }


    void set_descriptives(const vector<Descriptives>& new_descriptives)
    {
        descriptives = new_descriptives;
    }

    void set_min_max_range(const type& min, const type& max)
    {
        min_range = min;
        max_range = max;
    }


    void set_scalers(const vector<string>& new_scalers)
    {
        scalers = new_scalers;
    }

    void set_scalers(const string& new_scaler)
    {
        for (string& scaler : scalers)
            scaler = new_scaler;
    }

    void forward_propagate(const vector<TensorView>& input_views,
                           unique_ptr<LayerForwardPropagation>& forward_propagation,
                           const bool&) override
    {
        const Index outputs_number = get_outputs_number();

        ScalingForwardPropagation<Rank>* scaling_layer_forward_propagation =
            static_cast<ScalingForwardPropagation<Rank>*>(forward_propagation.get());

        const TensorMap2 inputs = tensor_map<2>(input_views[0]);

        Tensor2& outputs = scaling_layer_forward_propagation->outputs;
        outputs = inputs;

        for(Index i = 0; i < outputs_number; i++)
        {
            const string& scaler = scalers[i];

            if(scaler == "None")
                continue;
            else if(scaler == "MinimumMaximum")
                scale_minimum_maximum(outputs, i, descriptives[i], min_range, max_range);
            else if(scaler == "MeanStandardDeviation")
                scale_mean_standard_deviation(outputs, i, descriptives[i]);
            else if(scaler == "StandardDeviation")
                scale_standard_deviation(outputs, i, descriptives[i]);
            else if(scaler == "Logarithm")
                scale_logarithmic(outputs, i);
            else if(scaler == "ImageMinMax")
                outputs.chip(i,1).device(*device) =  outputs.chip(i,1) / type(255);
            else
                throw runtime_error("Unknown scaling method.\n");
        }
    }

    //void calculate_outputs(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>& );

    string write_no_scaling_expression(const vector<string>&, const vector<string>&) const;

    string write_minimum_maximum_expression(const vector<string>&, const vector<string>&) const;

    string write_mean_standard_deviation_expression(const vector<string>&, const vector<string>&) const;

    string write_standard_deviation_expression(const vector<string>&, const vector<string>&) const;

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

    void forward_propagate_cuda(const vector<float*>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

#endif

private:

    dimensions input_dimensions;

    vector<Descriptives> descriptives;

    vector<string> scalers;

    type min_range;
    type max_range;
};


template<int Rank>
struct ScalingForwardPropagation final : LayerForwardPropagation
{
    ScalingForwardPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }


    virtual ~ScalingForwardPropagation() = default;

    TensorView get_output_view() const override
    {
        const dimensions output_dimensions = layer->get_output_dimensions();

        return {(type*)outputs.data(), {batch_size, output_dimensions[0]}};
    }

    void initialize() override;

    void print() const override;

    Tensor2 outputs;
};


#ifdef OPENNN_CUDA

struct Scaling2dForwardPropagationCuda : public LayerForwardPropagationCuda
{
    Scaling2dForwardPropagationCuda(const Index & = 0, Layer* = nullptr);

    void set(const Index & = 0, Layer* = nullptr) override;

    void print() const override;

    void free() override;

    int* scalers_device = nullptr;
    type* minimums_device = nullptr;
    type* maximums_device = nullptr;
    type* means_device = nullptr;
    type* standard_deviations_device = nullptr;
};

#endif


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
