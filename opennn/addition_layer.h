//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tensors.h"

namespace opennn
{

template<int Rank> struct AdditionForwardPropagation;
template<int Rank> struct AdditionBackPropagation;

#ifdef OPENNN_CUDA
template<int Rank> struct AdditionForwardPropagationCuda;
template<int Rank> struct AdditionBackPropagationCuda;
#endif

template<int Rank>
class Addition final : public Layer
{

public:

    Addition(const dimensions& new_input_dimensions = {}, const string& new_name = "")
    {
        set(new_input_dimensions, new_name);
    }

    dimensions get_input_dimensions() const override
    {
        return input_dimensions;
    }

    dimensions get_output_dimensions() const override
    {
        return input_dimensions;
    }


    void set(const dimensions& new_input_dimensions, const string& new_label)
    {
        if(!new_input_dimensions.empty() && new_input_dimensions.size() != Rank)
            throw runtime_error("Input dimensions rank for AdditionLayer<" + to_string(Rank) + "> must be " + to_string(Rank));

        input_dimensions = new_input_dimensions;

        label = new_label;

        name = "Addition";
    }


    void forward_propagate(const vector<TensorView>& input_views,
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           const bool&) override
    {
        if (input_views.size() != 2)
            throw runtime_error(name + " layer requires exactly two inputs.");

        if (input_views[0].dims != input_views[1].dims)
            throw runtime_error("Input dimensions for " + name + " must be identical.");

        const TensorMap<Tensor<type, Rank>, Aligned16> input_1 = tensor_map<Rank>(input_views[0]);
        const TensorMap<Tensor<type, Rank>, Aligned16> input_2 = tensor_map<Rank>(input_views[1]);

        TensorMap<Tensor<type, Rank>, Aligned16> outputs = tensor_map<Rank>(layer_forward_propagation->outputs);

        outputs.device(*device) = input_1 + input_2;
    }


    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>& delta_views,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>& back_propagation) const override
    {
        if (delta_views.size() != 1)
            throw runtime_error(name + " backpropagation requires exactly one delta input.");

        const TensorMap<Tensor<type, Rank>, Aligned16> deltas = tensor_map<Rank>(delta_views[0]);

        TensorMap<Tensor<type, Rank>, Aligned16> input_deltas_0 = tensor_map<Rank>(back_propagation->input_deltas[0]);
        TensorMap<Tensor<type, Rank>, Aligned16> input_deltas_1 = tensor_map<Rank>(back_propagation->input_deltas[1]);

        input_deltas_0.device(*device) = deltas;
        input_deltas_1.device(*device) = deltas;
    }

    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* element = document.FirstChildElement("Addition");
        if(!element) throw runtime_error(name + " element is nullptr.");

        const string new_label = read_xml_string(element, "Label");
        const dimensions new_input_dimensions = string_to_dimensions(read_xml_string(element, "InputDimensions"));

        set(new_input_dimensions, new_label);
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement("Addition");

        add_xml_element(printer, "Label", label);
        add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));

        printer.CloseElement();
    }

#ifdef OPENNN_CUDA

public:

    void forward_propagate_cuda(const vector<TensorViewCuda>& inputs_device,
                                unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                const bool&) override
    {
        if (inputs_device.size() != 2)
            throw runtime_error(name + " layer requires exactly two inputs for CUDA propagation.");

        const size_t inputs_number = get_inputs_number();
        const size_t total_elements = static_cast<size_t>(forward_propagation_cuda->batch_size) * inputs_number;

        float alpha = 1.0f;
        float alpha_minus_one = -1.0f;
        const float beta = 0.0f;

        // @todo substitute addition_cuda by cudnn function similar as follows
/*
        cudnnOpTensor(cudnn_handle,
                      operator_sum_descriptor,
                      &alpha_minus_one,
                      output_tensor_descriptor,
                      targets,
                      &alpha,
                      output_tensor_descriptor,
                      outputs,
                      &beta,
                      output_tensor_descriptor,
                      errors_device);

*/
        addition_cuda(total_elements, inputs_device[0].data, inputs_device[1].data, forward_propagation_cuda->outputs.data);
    }


    void back_propagate_cuda(const vector<TensorViewCuda>&,
                             const vector<TensorViewCuda>& deltas_device,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const override
    {
        if (deltas_device.size() != 1)
            throw runtime_error(name + " backpropagation requires exactly one delta input for CUDA.");

        AdditionBackPropagationCuda<Rank>* this_back_propagation =
            static_cast<AdditionBackPropagationCuda<Rank>*>(back_propagation_cuda.get());

        const size_t inputs_number = get_inputs_number();
        const size_t total_elements = static_cast<size_t>(back_propagation_cuda->batch_size) * inputs_number;

        CHECK_CUDA(cudaMemcpy(this_back_propagation->input_deltas[0].data, deltas_device[0].data, total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(this_back_propagation->input_deltas[1].data, deltas_device[0].data, total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
    }

#endif

private:

    dimensions input_dimensions;
};


template<int Rank>
struct AdditionForwardPropagation final : LayerForwardPropagation
{
    AdditionForwardPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagation()
    {
        set(new_batch_size, new_layer);
    }


    void initialize() override
    {
        const dimensions output_dimensions = layer->get_output_dimensions();
        dimensions full_dims = { batch_size };
        full_dims.insert(full_dims.end(), output_dimensions.begin(), output_dimensions.end());

        outputs.dims = full_dims;
    }


    void print() const override
    {
        cout << "Addition Forward Propagation:" << endl;
        cout << "Outputs dimensions: " << outputs.dims << endl;
        cout << "Outputs data:" << endl << outputs.data << endl;
    }
};


template<int Rank>
struct AdditionBackPropagation final : LayerBackPropagation
{
    AdditionBackPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerBackPropagation()
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const dimensions input_dimensions = layer->get_input_dimensions();
        dimensions full_dims = { batch_size };
        full_dims.insert(full_dims.end(), input_dimensions.begin(), input_dimensions.end());

        input_deltas_memory.resize(2);
        input_deltas_memory[0].resize(count_elements(full_dims));
        input_deltas_memory[1].resize(count_elements(full_dims));

        input_deltas.resize(2);
        input_deltas[0].data = input_deltas_memory[0].data();
        input_deltas[0].dims = full_dims;
        input_deltas[1].data = input_deltas_memory[1].data();
        input_deltas[1].dims = full_dims;
    }


    void print() const override
    {
        cout << "Addition Back Propagation:" << endl;

        if(input_deltas.size() >= 1)
        {
            cout << "Input 1 Deltas dimensions: " << input_deltas[0].dims << endl;
            cout << input_deltas[0].data << endl;
        }

        if(input_deltas.size() >= 2)
        {
            cout << "Input 2 Deltas dimensions: " << input_deltas[1].dims << endl;
            cout << input_deltas[1].data << endl;
        }
    }
};


#ifdef OPENNN_CUDA

template<int Rank>
struct AdditionForwardPropagationCuda : public LayerForwardPropagationCuda
{
    AdditionForwardPropagationCuda(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const dimensions output_dimensions = layer->get_output_dimensions();
        dimensions full_dims = { static_cast<Index>(batch_size) };
        full_dims.insert(full_dims.end(), output_dimensions.begin(), output_dimensions.end());

        outputs.set_descriptor(full_dims);
    }

    void print() const override
    {
        // @todo
    }
};


template<int Rank>
struct AdditionBackPropagationCuda : public LayerBackPropagationCuda
{
    AdditionBackPropagationCuda(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerBackPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }


    void initialize() override
    {
        const dimensions input_dims = layer->get_input_dimensions();
        dimensions full_dims = { static_cast<Index>(batch_size) };
        full_dims.insert(full_dims.end(), input_dims.begin(), input_dims.end());

        input_deltas.resize(2);
        input_deltas[0].resize(full_dims);
        input_deltas[1].resize(full_dims);
    }


    void print() const override
    {
        // @todo
    }
};

#endif // OPENNN_CUDA

void reference_addition_layer();

} 

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
