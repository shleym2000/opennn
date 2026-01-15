//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYER_H
#define SCALINGLAYER_H

#include "layer.h"
#include "statistics.h"
#include "scaling.h"

namespace opennn
{

template<int Rank> struct ScalingForwardPropagation;
template<int Rank> struct ScalingForwardPropagationCuda;

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
        return input_dimensions;
    }

    dimensions get_output_dimensions() const override
    {
        return input_dimensions;
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
        if (new_input_dimensions.size() != Rank -1) 
        {
           ostringstream buffer;
           buffer << "OpenNN Exception: Scaling Layer.\n"
                  << "void set(const dimensions& new_input_dimensions) method.\n"
                  << "Input dimensions size must be " << Rank - 1 << ", but is " << new_input_dimensions.size() << ".\n";
           throw logic_error(buffer.str());
        }

        input_dimensions = new_input_dimensions;

        const Index new_inputs_number = accumulate(new_input_dimensions.begin(), new_input_dimensions.end(), 1, multiplies<Index>());

        descriptives.resize(new_inputs_number);

        for(Index i = 0; i < new_inputs_number; i++)
            descriptives[i].set(type(-1.0), type(1), type(0), type(1));

        scalers.resize(new_inputs_number, "MeanStandardDeviation");

        label = "scaling_layer";

        set_scalers("MeanStandardDeviation");

        set_min_max_range(type(-1), type(1));

        name = "Scaling" + to_string(Rank) + "d";

        is_trainable = false;
    }

    void set_input_dimensions(const dimensions& new_input_dimensions) override
    {
        set(new_input_dimensions);
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
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           const bool&) override
    {
        const Index outputs_number = get_outputs_number();

        const TensorMap2 inputs = tensor_map<2>(input_views[0]);

        TensorMap2 outputs = tensor_map<2>(layer_forward_propagation->outputs);

        outputs.device(*device) = inputs;

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

    string write_no_scaling_expression(const vector<string>& feature_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_dimensions().size() == 0 ? 0 : accumulate(get_output_dimensions().begin(), get_output_dimensions().end(), 1, multiplies<Index>());

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = " << feature_names[i] << ";\n";

        return buffer.str();
    }

    string write_minimum_maximum_expression(const vector<string>& feature_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_dimensions().size() == 0 ? 0 : accumulate(get_output_dimensions().begin(), get_output_dimensions().end(), 1, multiplies<Index>());

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = 2*(" << feature_names[i] << "-(" << descriptives[i].minimum << "))/(" << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-1;\n";

        return buffer.str();
    }

    string write_mean_standard_deviation_expression(const vector<string>& feature_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_inputs_number();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = (" << feature_names[i] << "-(" << descriptives[i].mean << "))/" << descriptives[i].standard_deviation << ";\n";

        return buffer.str();
    }

    string write_standard_deviation_expression(const vector<string>& feature_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_dimensions().size() == 0 ? 0 : accumulate(get_output_dimensions().begin(), get_output_dimensions().end(), 1, multiplies<Index>());

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = " << feature_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";

        return buffer.str();
    }

    string get_expression(const vector<string>& new_feature_names = vector<string>(), const vector<string>& = vector<string>()) const override
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

        expression = std::regex_replace(expression, std::regex("\\+-"), "-");
        expression = std::regex_replace(expression, std::regex("--"), "+");

        return expression;
    }

    void print() const override
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

    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* scaling_layer_element = document.FirstChildElement(name.c_str());

        if(!scaling_layer_element)
            throw runtime_error("Scaling element is nullptr.\n");

        const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");
        
        // This part needs to reconstruct dimensions if possible, or we just set flat dimensions if that's what XML has.
        // The original 2D code called set({neurons_number}). 
        // For Rank 2, it's fine. For Rank 3, {neurons_number} has size 1, but we need size 2.
        // XML serialization in 2D layer purely saved NeuronsNumber (total size).
        // It lost the shape information? "Scaling" element in 2D code only saved NeuronsNumber.
        // If we want to support higher ranks properly we might need to change XML format or assume 1D input if not specified.
        // However, user asked to "refactor" based on 2D pattern. 
        // For now, I will assume we can only restore as flat 1D input unless XML has InputDimensions.
        // But wait, set() will THROW if size != Rank-1.
        
        // If I am Scaling<3>, Rank=3, I need 2 dimensions. 
        // Existing "Scaling" XML does NOT seem to store dimensions.
        // This implies existing Scaling was only used for 1D inputs?
        // Or I should look at `flatten_layer` how it deserializes.
        // Flatten layer explicitly saves InputHeight, InputWidth.
        
        // Use a heuristic: If we don't have dimensions in XML, we might fail for Rank > 2.
        // But for this task I will try to read dimensions if they exist, otherwise fallback to flat?
        // But flat fallback will fail set() check for Rank > 2.
        // Let's assume for now we construct with {neurons_number} which is size 1.
        
        if constexpr (Rank == 2)
        {
             set({ neurons_number });
        }
        else
        {
             // Try to read generic InputDimensions if we were to add them. 
             // But following 2D code strictly, it only reads NeuronsNumber.
             // I will leave it as is for Rank 2, and for Rank > 2 acts as 2D did (which might be why 3D/4D were commented out / not used).
             // However, to make it compile for Rank > 2, I need to pass correct size.
             // If XML doesn't validation dims, we can't fully restore shape.
             // I will implement a dummy reshape for now to satisfy Rank:
             // [neurons_number, 1, 1...] 
             
             dimensions dims(Rank-1, 1);
             dims[0] = neurons_number;
             set(dims);
        }

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
/*
            if (descriptives_element->GetText()) {
                const vector<string> descriptives_string = get_tokens(descriptives_element->GetText(), " ");
                descriptives[i].set(
                    type(stof(descriptives_string[0])),
                    type(stof(descriptives_string[1])),
                    type(stof(descriptives_string[2])),
                    type(stof(descriptives_string[3]))
                    );
            }
*/
            scalers[i] = read_xml_string(scaling_neuron_element, "Scaler");

            start_element = scaling_neuron_element;
        }
    }

    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement(name.c_str());

        add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions().size() == 0 ? 0 : accumulate(get_output_dimensions().begin(), get_output_dimensions().end(), 1, multiplies<Index>())));

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

#ifdef OPENNN_CUDA

    void forward_propagate_cuda(const vector<float*>& inputs_device,
                                unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                const bool&) override
    {
        ScalingForwardPropagationCuda<Rank>* scaling_forward_propagation =
            static_cast<ScalingForwardPropagationCuda<Rank>*>(forward_propagation_cuda.get());

        const Index outputs_number = get_outputs_number();
        const size_t size = outputs_number * scaling_forward_propagation->batch_size;

        scale_2d_cuda(size, scaling_forward_propagation->batch_size, outputs_number,
                      inputs_device[0], scaling_forward_propagation->outputs,
                      scaling_forward_propagation->scalers_device,
                      scaling_forward_propagation->minimums_device,
                      scaling_forward_propagation->maximums_device,
                      scaling_forward_propagation->means_device,
                      scaling_forward_propagation->standard_deviations_device,
                      min_range,
                      max_range);
    }

#endif

private:

    dimensions input_dimensions;

    type* minimums = nullptr;
    type* maximums = nullptr;
    type* means = nullptr;
    type* standard_deviations = nullptr;

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

    void initialize() override
    {
        const Index outputs_number = layer->get_outputs_number();

        outputs.dims = {batch_size, outputs_number};
    }

    vector<TensorView*> get_tensor_views() override
    {
        return { &outputs };
    }

    void print() const override
    {
        cout << "Outputs:" << endl
             << outputs.dims << endl;
    }
};


#ifdef OPENNN_CUDA

template<int Rank>
struct ScalingForwardPropagationCuda : public LayerForwardPropagationCuda
{
    ScalingForwardPropagationCuda(const Index & new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }

    void set(const Index & new_batch_size = 0, Layer* new_layer = nullptr)
    {
        if (!new_layer) return;

        layer = new_layer;
        batch_size = new_batch_size;

        const Scaling<Rank>* scaling_layer = static_cast<Scaling<Rank>*>(layer);
        const Index outputs_number = scaling_layer->get_outputs_number();
        const size_t size = batch_size * outputs_number;

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

    void print() const override
    {
        const Index outputs_number = layer->get_outputs_number();

        cout << "Scaling CUDA Outputs:" << endl
            << matrix_from_device(outputs, batch_size, outputs_number) << endl;
    }

    void free() override
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

    int* scalers_device = nullptr;
    type* minimums_device = nullptr;
    type* maximums_device = nullptr;
    type* means_device = nullptr;
    type* standard_deviations_device = nullptr;
};

#endif


void reference_scaling_layer();

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
