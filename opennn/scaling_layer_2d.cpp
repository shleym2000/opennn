//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "strings_utilities.h"
#include "tensors.h"
#include "statistics.h"
#include "scaling_layer_2d.h"
#include "scaling.h"

namespace opennn
{
/*
template<int Rank>
Scaling<Rank>::Scaling(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


template<int Rank>
dimensions Scaling<Rank>::get_input_dimensions() const
{
    return dimensions{Index(scalers.size())};
}


template<int Rank>
dimensions Scaling<Rank>::get_output_dimensions() const
{
    return dimensions{Index(scalers.size())};
}


template<int Rank>
vector<Descriptives> Scaling<Rank>::get_descriptives() const
{
    return descriptives;
}


template<int Rank>
Descriptives Scaling<Rank>::get_descriptives(const Index& index) const
{
    return descriptives[index];
}


template<int Rank>
Tensor1 Scaling<Rank>::get_minimums() const
{
    const Index outputs_number = get_outputs_number();

    Tensor1 minimums(outputs_number);

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        minimums[i] = descriptives[i].minimum;

    return minimums;
}


template<int Rank>
Tensor1 Scaling<Rank>::get_maximums() const
{
    const Index outputs_number = get_outputs_number();

    Tensor1 maximums(outputs_number);

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        maximums[i] = descriptives[i].maximum;

    return maximums;
}


template<int Rank>
Tensor1 Scaling<Rank>::get_means() const
{
    const Index outputs_number = get_outputs_number();

    Tensor1 means(outputs_number);

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        means[i] = descriptives[i].mean;

    return means;
}


template<int Rank>
Tensor1 Scaling<Rank>::get_standard_deviations() const
{
    const Index outputs_number = get_outputs_number();

    Tensor1 standard_deviations(outputs_number);

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        standard_deviations[i] = descriptives[i].standard_deviation;

    return standard_deviations;
}


template<int Rank>
vector<string> Scaling<Rank>::get_scalers() const
{
    return scalers;
}


template<int Rank>
void Scaling<Rank>::set(const dimensions& new_input_dimensions)
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


template<int Rank>
void Scaling<Rank>::set_input_dimensions(const dimensions& new_input_dimensions)
{
    descriptives.resize(new_input_dimensions[0]);

    scalers.resize(new_input_dimensions[0], "MeanStandardDeviation");
}


template<int Rank>
void Scaling<Rank>::set_output_dimensions(const dimensions& new_output_dimensions)
{
    set_input_dimensions(new_output_dimensions);
}


template<int Rank>
void Scaling<Rank>::set_min_max_range(const type& min, const type& max)
{
    min_range = min;
    max_range = max;
}


template<int Rank>
void Scaling<Rank>::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    descriptives = new_descriptives;
}


template<int Rank>
void Scaling<Rank>::set_scalers(const vector<string>& new_scalers)
{
    scalers = new_scalers;
}


template<int Rank>
void Scaling<Rank>::set_scalers(const string& new_scaler)
{
    for (string& scaler : scalers)
        scaler = new_scaler;
}


template<int Rank>
void Scaling<Rank>::forward_propagate(const vector<TensorView>& input_views,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  const bool&)
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


template<int Rank>
void Scaling<Rank>::calculate_outputs(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                  type* outputs_data, const Tensor<Index, 1>& outputs_dimensions)
{
    const Index input_rank = inputs_dimensions.size();

    if(input_rank == 2)
    {
        const Index points_number = inputs_dimensions(0);
        const Index neurons_number = get_inputs_number();

        const Tensor<Index, 0> input_size = inputs_dimensions.prod();

        const TensorMap2 inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));
        TensorMap2 outputs(outputs_data, outputs_dimensions[0], outputs_dimensions(1));

        if(outputs_dimensions[0] != points_number || outputs_dimensions(1) != neurons_number)
            throw runtime_error("Outputs dimensions must be equal");

        for(Index i = 0; i < neurons_number; i++)
        {
            const string scaler = scalers[i];

            Tensor1 column = inputs.chip(i, 1);

            if(scaler == "None")
                column = inputs.chip(i,1);
            else if(scaler == "MinimumMaximum")
                column = (inputs.chip(i, 1) - descriptives[i].minimum) / (descriptives[i].maximum - descriptives[i].minimum);
            else if(scaler == "MeanStandardDeviation")
                column = (inputs.chip(i, 1) - descriptives[i].mean) / descriptives[i].standard_deviation;
            else if(scaler == "StandardDeviation")
                column = (1/descriptives[i].standard_deviation) * inputs.chip(i, 1);
            else if(scaler == "Logarithm")
                column = inputs.chip(i,1).log();
            else
                throw runtime_error("Unknown scaling method.\n");

            outputs.chip(i, 1) = column;
        }
    }
    else if(input_rank == 4)
    {
        const Tensor<bool, 0> equal_dimensions = (inputs_dimensions == outputs_dimensions).any().all();

        if(!equal_dimensions(0))
        {
            throw runtime_error("Input and output data must have the same dimensions.\n");
        }

        TensorMap4 input(inputs_data, inputs_dimensions(0), inputs_dimensions(1), inputs_dimensions(2), inputs_dimensions(3));

        TensorMap4 output(outputs_data, inputs_dimensions(0), inputs_dimensions(1), inputs_dimensions(2), inputs_dimensions(3));

        for(Index i = 0; i < input.size(); i++)
            output(i) = -static_cast<type>(1) + static_cast<type>(2*input(i)/255);
    }
    else
        throw runtime_error("Input dimension must be 2 or 4.\n");
}
*/

template<int Rank>
string Scaling<Rank>::write_no_scaling_expression(const vector<string>& feature_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = " << feature_names[i] << ";\n";

    return buffer.str();
}


template<int Rank>
string Scaling<Rank>::write_minimum_maximum_expression(const vector<string>& feature_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = 2*(" << feature_names[i] << "-(" << descriptives[i].minimum << "))/(" << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-1;\n";

    return buffer.str();
}


template<int Rank>
string Scaling<Rank>::write_mean_standard_deviation_expression(const vector<string>& feature_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_inputs_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = (" << feature_names[i] << "-(" << descriptives[i].mean << "))/" << descriptives[i].standard_deviation << ";\n";

    return buffer.str();
}


template<int Rank>
string Scaling<Rank>::write_standard_deviation_expression(const vector<string>& feature_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = " << feature_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";

    return buffer.str();
}


template<int Rank>
string Scaling<Rank>::get_expression(const vector<string>& new_feature_names, const vector<string>&) const
{
    const vector<string> feature_names = new_feature_names.empty()
                                           ? get_default_feature_names()
                                           : new_feature_names;

    const Index outputs_number = get_outputs_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < outputs_number; i++)
    {
        const string& scaler = scalers[i];

        if(scaler == "None")
            buffer << "scaled_" << feature_names[i] << " = " << feature_names[i] << ";\n";
        else if(scaler == "MinimumMaximum")
            buffer << "scaled_" << feature_names[i]
                   << " = " << feature_names[i] << "*(" << max_range << "-" << min_range << ")/("
                   << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-" << descriptives[i].minimum << "*("
                   << max_range << "-" << min_range << ")/("
                   << descriptives[i].maximum << "-" << descriptives[i].minimum << ")+" << min_range << ";\n";
        else if(scaler == "MeanStandardDeviation")
            buffer << "scaled_" << feature_names[i] << " = (" << feature_names[i] << "-" << descriptives[i].mean << ")/" << descriptives[i].standard_deviation << ";\n";
        else if(scaler == "StandardDeviation")
            buffer << "scaled_" << feature_names[i] << " = " << feature_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";
        else if(scaler == "Logarithm")
            buffer << "scaled_" << feature_names[i] << " = log(" << feature_names[i] << ");\n";
        else
            throw runtime_error("Unknown inputs scaling method.\n");
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}


template<int Rank>
void Scaling<Rank>::print() const
{
    cout << "Scaling layer" << endl;

    const Index inputs_number = get_inputs_number();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Neuron " << i << endl
             << "string " << scalers[i] << endl;

        descriptives[i].print();
    }
}


template<int Rank>
void Scaling<Rank>::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Scaling");

    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));

    const Index outputs_number = get_outputs_number();

    for (Index i = 0; i < outputs_number; i++)
    {
        printer.OpenElement("ScalingNeuron");
        printer.PushAttribute("Index", int(i + 1));
        add_xml_element(printer, "Descriptives", tensor_to_string<type, 1>(descriptives[i].to_tensor()));
        add_xml_element(printer, "Scaler", scalers[i]);

        printer.CloseElement();
    }

    printer.CloseElement();
}


template<int Rank>
void Scaling<Rank>::from_XML(const XMLDocument& document)
{
    const XMLElement* scaling_layer_element = document.FirstChildElement("Scaling");

    if(!scaling_layer_element)
        throw runtime_error("Scaling element is nullptr.\n");

    const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");
    set({ neurons_number });

    const XMLElement* start_element = scaling_layer_element->FirstChildElement("NeuronsNumber");

    for (Index i = 0; i < neurons_number; i++) {
        const XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
        if (!scaling_neuron_element) {
            throw runtime_error("Scaling neuron " + to_string(i + 1) + " is nullptr.\n");
        }

        unsigned index = 0;
        scaling_neuron_element->QueryUnsignedAttribute("Index", &index);
        if (index != i + 1) {
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");
        }

        const XMLElement* descriptives_element = scaling_neuron_element->FirstChildElement("Descriptives");

        if (!descriptives_element)
            throw runtime_error("Descriptives element " + to_string(i + 1) + " is nullptr.\n");

        if (descriptives_element->GetText()) {
            const vector<string> descriptives_string = get_tokens(descriptives_element->GetText(), " ");
            descriptives[i].set(
                type(stof(descriptives_string[0])),
                type(stof(descriptives_string[1])),
                type(stof(descriptives_string[2])),
                type(stof(descriptives_string[3]))
                );
        }

        scalers[i] = read_xml_string(scaling_neuron_element, "Scaler");

        start_element = scaling_neuron_element;
    }
}

/*
template<int Rank>
ScalingForwardPropagation<Rank>::ScalingForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


template<int Rank>
TensorView ScalingForwardPropagation<Rank>::get_output_view() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return {(type*)outputs.data(), {batch_size, output_dimensions[0]}};
}
*/

template<int Rank>
void ScalingForwardPropagation<Rank>::initialize()
{
    const Index outputs_number = layer->get_outputs_number();

    outputs.resize(batch_size, outputs_number);
}


template<int Rank>
void ScalingForwardPropagation<Rank>::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}


#ifdef  OPENNN_CUDA

void Scaling::forward_propagate_cuda(const vector<float*>& inputs_device,
                                       unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                       const bool&)
{
    ScalingForwardPropagationCuda* scaling_2d_forward_propagation =
        static_cast<ScalingForwardPropagationCuda*>(forward_propagation_cuda.get());

    const Index outputs_number = get_outputs_number();
    const size_t size = outputs_number * scaling_2d_forward_propagation->batch_size;

    scale_2d_cuda(size, scaling_2d_forward_propagation->batch_size, outputs_number,
                  inputs_device[0], scaling_2d_forward_propagation->outputs,
                  scaling_2d_forward_propagation->scalers_device,
                  scaling_2d_forward_propagation->minimums_device,
                  scaling_2d_forward_propagation->maximums_device,
                  scaling_2d_forward_propagation->means_device,
                  scaling_2d_forward_propagation->standard_deviations_device,
                  min_range,
                  max_range);
}

// CUDA structs

ScalingForwardPropagationCuda::ScalingForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ScalingForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;
    batch_size = new_batch_size;

    const Scaling* scaling_layer = static_cast<Scaling*>(layer);
    const Index outputs_number = scaling_layer->get_outputs_number();
    const size_t size = batch_size * outputs_number;

    //CUDA_MALLOC_AND_REPORT(outputs, size * sizeof(float));
    CHECK_CUDA(cudaMalloc(&outputs, size * sizeof(float)));
    
    const Tensor1 minimums_host = scaling_layer->get_minimums();
    const Tensor1 maximums_host = scaling_layer->get_maximums();
    const Tensor1 means_host = scaling_layer->get_means();
    const Tensor1 std_devs_host = scaling_layer->get_standard_deviations();
    const vector<string> scalers_host_vec = scaling_layer->get_scalers();

    Tensor<int, 1> scalers_host_tensor(outputs_number);
    for (Index i = 0; i < outputs_number; ++i)
    {
        const string & scaler_str = scalers_host_vec[i];

        if (scaler_str == "None")
            scalers_host_tensor(i) = 0;
        else if (scaler_str == "MinimumMaximum")
            scalers_host_tensor(i) = 1;
        else if (scaler_str == "MeanStandardDeviation")
            scalers_host_tensor(i) = 2;
        else if (scaler_str == "StandardDeviation")
            scalers_host_tensor(i) = 3;
        else if (scaler_str == "Logarithm")
            scalers_host_tensor(i) = 4;
        else if (scaler_str == "ImageMinMax")
            scalers_host_tensor(i) = 5;
        else
            throw runtime_error("Unknown scaler method for CUDA: " + scaler_str);
    }

    //CUDA_MALLOC_AND_REPORT(minimums_device, outputs_number * sizeof(float));
    //CUDA_MALLOC_AND_REPORT(maximums_device, outputs_number * sizeof(float));
    //CUDA_MALLOC_AND_REPORT(means_device, outputs_number * sizeof(float));
    //CUDA_MALLOC_AND_REPORT(standard_deviations_device, outputs_number * sizeof(float));
    //CUDA_MALLOC_AND_REPORT(scalers_device, outputs_number * sizeof(int));
    CHECK_CUDA(cudaMalloc(&minimums_device, outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&maximums_device, outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&means_device, outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&standard_deviations_device, outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&scalers_device, outputs_number * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(minimums_device, minimums_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(maximums_device, maximums_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(means_device, means_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(standard_deviations_device, std_devs_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(scalers_device, scalers_host_tensor.data(), outputs_number * sizeof(int), cudaMemcpyHostToDevice));
}


void ScalingForwardPropagationCuda::print() const
{
    const Index outputs_number = layer->get_outputs_number();

    cout << "Scaling CUDA Outputs:" << endl
        << matrix_from_device(outputs, batch_size, outputs_number) << endl;
}


void ScalingForwardPropagationCuda::free()
{
    cudaFree(outputs);
    cudaFree(scalers_device);
    cudaFree(minimums_device);
    cudaFree(maximums_device);
    cudaFree(means_device);
    cudaFree(standard_deviations_device);

    outputs = nullptr;
    scalers_device = nullptr;
    minimums_device = nullptr;
    maximums_device = nullptr;
    means_device = nullptr;
    standard_deviations_device = nullptr;
}

REGISTER(LayerForwardPropagationCuda, ScalingForwardPropagationCuda, "Scaling")

#endif

using Scaling2d = Scaling<2>;
using Scaling3d = Scaling<3>;
using Scaling4d = Scaling<4>;

using ScalingForwardPropagation2d = ScalingForwardPropagation<2>;
using ScalingForwardPropagation3d = ScalingForwardPropagation<3>;
using ScalingForwardPropagation4d = ScalingForwardPropagation<4>;

REGISTER(Layer, Scaling2d, "Scaling2d")
REGISTER(Layer, Scaling3d, "Scaling3d")
REGISTER(Layer, Scaling4d, "Scaling4d")

REGISTER(LayerForwardPropagation, ScalingForwardPropagation2d, "Scaling2d")
REGISTER(LayerForwardPropagation, ScalingForwardPropagation3d, "Scaling3d")
REGISTER(LayerForwardPropagation, ScalingForwardPropagation4d, "Scaling4d")

//REGISTER(Layer, Scaling, "Scaling")
//REGISTER(LayerForwardPropagation, ScalingForwardPropagation, "Scaling")

}

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
