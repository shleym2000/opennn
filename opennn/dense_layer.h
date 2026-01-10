//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef DENSE2D_H
#define DENSE2D_H

#include "layer.h"

namespace opennn
{

template<int Rank>
struct DenseForwardPropagation final : LayerForwardPropagation
{
    DenseForwardPropagation(const Index& = 0, Layer* = nullptr);

    virtual ~DenseForwardPropagation() = default;

    TensorView get_output_view() const override;

    void initialize() override;

    void print() const override;

    Tensor1 means;
    Tensor1 standard_deviations;
    Tensor2 normalized_outputs;

    Tensor2 outputs;

    Tensor2 activation_derivatives;
};

template<int Rank>
struct DenseBackPropagation final : LayerBackPropagation
{
    DenseBackPropagation(const Index& = 0, Layer* = nullptr);
    virtual ~DenseBackPropagation() = default;

    vector<TensorView> get_input_derivative_views() const override;

    vector<ParameterView> get_parameter_delta_views() const override;

    void initialize() override;

    void print() const override;

    Tensor2 input_deltas;

    Tensor1 bias_deltas;
    Tensor2 weight_deltas;

    Tensor1 bn_scale_deltas;
    Tensor1 bn_offset_deltas;
};


struct Dense2dBackPropagationLM : LayerBackPropagationLM
{
    Dense2dBackPropagationLM(const Index& = 0, Layer* = nullptr);

    vector<TensorView> get_input_derivative_views() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor2 input_deltas;

    Tensor2 squared_errors_Jacobian;
};


struct Dense2dForwardPropagationLM;

template<int Rank>
class Dense final : public Layer
{

public:

    Dense(const dimensions& new_input_dimensions = {0},
            const dimensions& new_output_dimensions = {0},
            const string& new_activation_function = "HyperbolicTangent",
            const bool& new_batch_normalization = false,
            const string& new_label = "dense2d_layer")
    {
        set(new_input_dimensions, new_output_dimensions, new_activation_function, new_batch_normalization, new_label);
    }

    dimensions get_input_dimensions() const override
    {
        return { weights.dimension(0) };
    }

    dimensions get_output_dimensions() const override
    {
        return { biases.size() };
    }

    vector<ParameterView> get_parameter_views() const override
    {
        vector<ParameterView> parameter_views = {{(type*)(biases.data()), biases.size()},
                                                 {(type*)(weights.data()), weights.size()}};

        if (batch_normalization)
        {
            parameter_views.push_back({ const_cast<type*>(scales.data()), scales.size() });
            parameter_views.push_back({ const_cast<type*>(offsets.data()), offsets.size() });
        }

        return parameter_views;
    }

    type get_dropout_rate() const
    {
        return dropout_rate;
    }

    bool get_batch_normalization() const
    {
        return batch_normalization;
    }

    Tensor1 get_scales() const
    {
        return scales;
    }

    Tensor1 get_offsets() const
    {
        return offsets;
    }

    const string& get_activation_function() const
    {
        return activation_function;
    }

    void set(const dimensions& new_input_dimensions = {},
             const dimensions& new_output_dimensions = {},
             const string& new_activation_function = "HyperbolicTangent",
             const bool& new_batch_normalization = false,
             const string& new_label = "dense2d_layer")
    {
        if (new_input_dimensions.size() != 1)
            throw runtime_error("Input dimensions size is not 1");

        if (new_output_dimensions.size() != 1)
            throw runtime_error("Output dimensions size is not 1");

        biases.resize(new_output_dimensions[0]);
        weights.resize(new_input_dimensions[0], new_output_dimensions[0]);

        set_parameters_random();

        set_activation_function(new_activation_function);

        set_batch_normalization(new_batch_normalization);

        const Index outputs_number = get_outputs_number();

        if (batch_normalization)
        {
            scales.resize(outputs_number);
            scales.setConstant(1.0);

            offsets.resize(outputs_number);
            offsets.setZero();

            moving_means.resize(outputs_number);
            moving_means.setZero();

            moving_standard_deviations.resize(outputs_number);
            moving_standard_deviations.setZero();
        }

        set_label(new_label);

        name = "Dense2d";

#ifdef OPENNN_CUDA

        if (batch_normalization)
        {
            cudnnCreateTensorDescriptor(&bn_tensor_descriptor);

            cudnnSetTensor4dDescriptor(bn_tensor_descriptor,
                                       CUDNN_TENSOR_NCHW,
                                       CUDNN_DATA_FLOAT,
                                       1, outputs_number, 1, 1);
        }

#endif
    }

    void set_input_dimensions(const dimensions& new_input_dimensions) override
    {
        const Index inputs_number = new_input_dimensions[0];
        const Index outputs_number = get_outputs_number();

        biases.resize(outputs_number);

        weights.resize(inputs_number, outputs_number);
    }

    void set_output_dimensions(const dimensions& new_output_dimensions) override
    {
        const Index inputs_number = get_inputs_number();
        const Index neurons_number = new_output_dimensions[0];

        biases.resize(neurons_number);

        weights.resize(inputs_number, neurons_number);
    }


    void set_activation_function(const string& new_activation_function)
    {
        static const unordered_set<string> activation_functions =
            {"Logistic", "HyperbolicTangent", "Linear", "RectifiedLinear", "ScaledExponentialLinear", "Softmax"};

        if(activation_functions.count(new_activation_function))
            activation_function = new_activation_function;
        else
            throw runtime_error("Unknown activation function: " + new_activation_function);

#ifdef OPENNN_CUDA

        if (activation_descriptor == nullptr && activation_function != "Softmax")
            cudnnCreateActivationDescriptor(&activation_descriptor);

        cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY;
        double relu_ceiling = 0.0;

        if (activation_function == "Linear")
        {
            activation_mode = CUDNN_ACTIVATION_IDENTITY;
            use_combinations = false;
        }
        else if (activation_function == "Logistic")
        {
            activation_mode = CUDNN_ACTIVATION_SIGMOID;
            use_combinations = false;
        }
        else if (activation_function == "HyperbolicTangent")
        {
            activation_mode = CUDNN_ACTIVATION_TANH;
            use_combinations = false;
        }
        else if (activation_function == "RectifiedLinear")
        {
            activation_mode = CUDNN_ACTIVATION_RELU;
            use_combinations = false;
        }
        else if (activation_function == "ScaledExponentialLinear")
        {
            activation_mode = CUDNN_ACTIVATION_ELU;
            use_combinations = true;
        }
        else if (activation_function == "ClippedRelu")
        {
            activation_mode = CUDNN_ACTIVATION_CLIPPED_RELU;
            use_combinations = true;
            relu_ceiling = 6.0;
        }
        else if (activation_function == "Swish")
        {
            activation_mode = CUDNN_ACTIVATION_SWISH;
            use_combinations = true;
        }
        else if (activation_function == "Softmax")
            use_combinations = true;

        if (activation_function != "Softmax")
            cudnnSetActivationDescriptor(activation_descriptor, activation_mode, CUDNN_PROPAGATE_NAN, relu_ceiling);

#endif
    }

    void set_dropout_rate(const type&);

    void normalization(Tensor1&, Tensor1&, const Tensor2&, Tensor2&) const;

    void set_batch_normalization(const bool&);

    void apply_batch_normalization_backward(TensorMap2& deltas,
                                            unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                            unique_ptr<LayerBackPropagation>& layer_back_propagation) const
    {
        const DenseForwardPropagation<2>* dense2d_forward_propagation =
            static_cast<const DenseForwardPropagation<2>*>(layer_forward_propagation.get());

        const Index batch_size = dense2d_forward_propagation->batch_size;

        const Tensor2& normalized_outputs = dense2d_forward_propagation->normalized_outputs;
        const Tensor1& standard_deviations = dense2d_forward_propagation->standard_deviations;

        DenseBackPropagation<2>* dense2d_back_propagation =
            static_cast<DenseBackPropagation<2>*>(layer_back_propagation.get());

        Tensor1& bn_scale_deltas = dense2d_back_propagation->bn_scale_deltas;
        Tensor1& bn_offset_deltas = dense2d_back_propagation->bn_offset_deltas;

        const array<int, 1> reduction_axes = { 0 };
        const array<Index, 2> reshape_dims = { 1, get_outputs_number() };
        const array<Index, 2> broadcast_dims = { batch_size, 1 };

        bn_offset_deltas.device(*device) = deltas.sum(reduction_axes);
        bn_scale_deltas.device(*device) = (deltas * normalized_outputs).sum(reduction_axes);

        const auto inv_m = type(1) / batch_size;

        deltas.device(*device) =
            ((deltas * type(batch_size))
             - bn_offset_deltas.reshape(reshape_dims).broadcast(broadcast_dims)
             - normalized_outputs *
                   bn_scale_deltas.reshape(reshape_dims).broadcast(broadcast_dims)
             ) * inv_m
            / standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims)
            * scales.reshape(reshape_dims).broadcast(broadcast_dims);
    }

    void forward_propagate(const vector<TensorView>& input_views,
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           const bool& is_training) override
    {
        const TensorMap2 inputs = tensor_map<2>(input_views[0]);

        DenseForwardPropagation<2>* dense2d_forward_propagation =
            static_cast<DenseForwardPropagation<2>*>(layer_forward_propagation.get());

        Tensor2& outputs = dense2d_forward_propagation->outputs;

        calculate_combinations<2>(inputs, weights, biases, outputs);

        if(batch_normalization)
            normalize_batch<2>(
                dense2d_forward_propagation->outputs,
                dense2d_forward_propagation->normalized_outputs,
                dense2d_forward_propagation->means,
                dense2d_forward_propagation->standard_deviations,
                moving_means,
                moving_standard_deviations,
                scales,
                offsets,
                is_training);

        is_training
            ? calculate_activations(activation_function, outputs, dense2d_forward_propagation->activation_derivatives)
            : calculate_activations(activation_function, outputs, empty_2);

        if(is_training && dropout_rate > type(0))
            dropout(outputs, dropout_rate);
    }


    void back_propagate(const vector<TensorView>& input_views,
                        const vector<TensorView>& delta_views,
                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                        unique_ptr<LayerBackPropagation>& back_propagation) const override
    {
        const TensorMap2 inputs = tensor_map<2>(input_views[0]);
        TensorMap2 deltas = tensor_map<2>(delta_views[0]);

        // Forward propagation

        const DenseForwardPropagation<2>* dense2d_layer_forward_propagation =
            static_cast<DenseForwardPropagation<2>*>(forward_propagation.get());

        const Tensor2& activation_derivatives = dense2d_layer_forward_propagation->activation_derivatives;

        // Back propagation

        DenseBackPropagation<2>* dense2d_back_propagation =
            static_cast<DenseBackPropagation<2>*>(back_propagation.get());

        Tensor2& weight_deltas = dense2d_back_propagation->weight_deltas;

        Tensor1& bias_deltas = dense2d_back_propagation->bias_deltas;

        const bool& is_first_layer = dense2d_back_propagation->is_first_layer;

        Tensor2& input_deltas = dense2d_back_propagation->input_deltas;

        if(activation_function != "Softmax")
            deltas.device(*device) = deltas * activation_derivatives;

        if (batch_normalization)
            apply_batch_normalization_backward(deltas, forward_propagation, back_propagation);

        bias_deltas.device(*device) = deltas.sum(array_1(0));

        weight_deltas.device(*device) = inputs.contract(deltas, axes(0,0));

        if (!is_first_layer)
            input_deltas.device(*device) = deltas.contract(weights, axes(1,1));
    }

    void back_propagate_lm(const vector<TensorView>& input_views,
                           const vector<TensorView>& delta_views,
                           unique_ptr<LayerForwardPropagation>& forward_propagation,
                           unique_ptr<LayerBackPropagationLM>& back_propagation) const override
    {
        const TensorMap2 inputs = tensor_map<2>(input_views[0]);
        TensorMap2 deltas = tensor_map<2>(delta_views[0]);

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        const Index biases_number = biases.size();

        // Forward propagation

        const DenseForwardPropagation<2>* dense2d_layer_forward_propagation =
            static_cast<DenseForwardPropagation<2>*>(forward_propagation.get());

        const Tensor2& activation_derivatives
            = dense2d_layer_forward_propagation->activation_derivatives;

        // Back propagation

        Dense2dBackPropagationLM* dense2d_layer_back_propagation_lm =
            static_cast<Dense2dBackPropagationLM*>(back_propagation.get());

        Tensor2& squared_errors_Jacobian = dense2d_layer_back_propagation_lm->squared_errors_Jacobian;

        const bool& is_first_layer = dense2d_layer_back_propagation_lm->is_first_layer;

        Tensor2& input_deltas = dense2d_layer_back_propagation_lm->input_deltas;

        deltas.device(*device) = deltas * activation_derivatives;

        squared_errors_Jacobian.slice(array<Index, 2>{0, 0}, array<Index, 2>{deltas.dimension(0), biases_number})
            .device(*device) = deltas;

        for (Index j = 0; j < outputs_number; j++)
        {
            const Tensor1 delta_j = deltas.chip(j, 1);

            for (Index i = 0; i < inputs_number; i++)
            {
                const Tensor1 input_i = inputs.chip(i, 1);

                const Tensor1 derivative = delta_j * input_i;

                const Index weight_column_index = biases_number + (j * inputs_number) + i;

                squared_errors_Jacobian.chip(weight_column_index, 1)
                    .device(*device) = derivative;
            }
        }

        if (!is_first_layer)
        {
            const Tensor2 weights_transposed = weights.shuffle(array<int, 2>{1, 0});
            input_deltas.device(*device) = deltas.contract(weights_transposed, axes(1, 0));
        }
    }

    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                           const Index& index,
                                           Tensor2& squared_errors_Jacobian) const override
    {
        const Index parameters_number = get_parameters_number();
        const Index batch_size = back_propagation->batch_size;

        Dense2dBackPropagationLM* dense2d_layer_back_propagation_lm =
            static_cast<Dense2dBackPropagationLM*>(back_propagation.get());

        type* this_squared_errors_Jacobian_data = dense2d_layer_back_propagation_lm->squared_errors_Jacobian.data();

        memcpy(squared_errors_Jacobian.data() + index,
               this_squared_errors_Jacobian_data,
               parameters_number * batch_size * sizeof(type));
    }

    string get_expression(const vector<string>& new_feature_names = vector<string>(), const vector<string>& new_output_names = vector<string>()) const override
    {
        const vector<string> feature_names = new_feature_names.empty()
        ? get_default_feature_names()
        : new_feature_names;

        const vector<string> output_names = new_output_names.empty()
                                                ? get_default_output_names()
                                                : new_output_names;

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        ostringstream buffer;

        for(Index j = 0; j < outputs_number; j++)
        {
            const TensorMap1 weights_column = tensor_map(weights, j);

            buffer << output_names[j] << " = " << activation_function << "( " << biases(j) << " + ";

            for(Index i = 0; i < inputs_number - 1; i++)
                buffer << "(" << weights_column(i) << "*" << feature_names[i] << ") + ";

            buffer << "(" << weights_column(inputs_number - 1) << "*" << feature_names[inputs_number - 1] << ") );\n";
        }

        return buffer.str();
    }

    void print() const override
    {
        cout << "Dense2d layer" << endl
             << "Input dimensions: " << get_input_dimensions()[0] << endl
             << "Output dimensions: " << get_output_dimensions()[0] << endl
             << "Biases dimensions: " << biases.dimensions() << endl
             << "Weights dimensions: " << weights.dimensions() << endl;

        cout << "Activation function:" << endl;
        cout << activation_function << endl;
    }

    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* dense2d_layer_element = document.FirstChildElement("Dense2d");

        if(!dense2d_layer_element)
            throw runtime_error("Dense2d element is nullptr.\n");

        set_label(read_xml_string(dense2d_layer_element, "Label"));

        const Index inputs_number = read_xml_index(dense2d_layer_element, "InputsNumber");
        const Index neurons_number = read_xml_index(dense2d_layer_element, "NeuronsNumber");

        set_input_dimensions({ inputs_number });
        set_output_dimensions({ neurons_number });

        set_activation_function(read_xml_string(dense2d_layer_element, "Activation"));

        bool use_batch_normalization = false;
        const XMLElement* bn_element = dense2d_layer_element->FirstChildElement("BatchNormalization");
        if (bn_element && bn_element->GetText())
            use_batch_normalization = (string(bn_element->GetText()) == "true");
        set_batch_normalization(use_batch_normalization);
        if (batch_normalization)
        {
            scales.resize(neurons_number);
            offsets.resize(neurons_number);
            moving_means.resize(neurons_number);
            moving_standard_deviations.resize(neurons_number);

            string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "Scales"), scales);
            string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "Offsets"), offsets);
            string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "MovingMeans"), moving_means);
            string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "MovingStandardDeviations"), moving_standard_deviations);
        }

        string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "Biases"), biases);
        string_to_tensor<type, 2>(read_xml_string(dense2d_layer_element, "Weights"), weights);
    }

    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement("Dense2d");

        add_xml_element(printer, "Label", label);
        add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
        add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
        add_xml_element(printer, "Activation", activation_function);
        add_xml_element(printer, "BatchNormalization", batch_normalization ? "true" : "false");

        if (batch_normalization)
        {
            add_xml_element(printer, "Scales", tensor_to_string<type, 1>(scales));
            add_xml_element(printer, "Offsets", tensor_to_string<type, 1>(offsets));
            add_xml_element(printer, "MovingMeans", tensor_to_string<type, 1>(moving_means));
            add_xml_element(printer, "MovingStandardDeviations", tensor_to_string<type, 1>(moving_standard_deviations));
        }

        add_xml_element(printer, "Biases", tensor_to_string<type, 1>(biases));
        add_xml_element(printer, "Weights", tensor_to_string<type, 2>(weights));

        printer.CloseElement();
    }

#ifdef OPENNN_CUDA

public:

    void forward_propagate_cuda(const vector<float*>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

    void back_propagate_cuda(const vector<float*>&,
                             const vector<float*>&,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>&) const override;

    vector<ParameterView> get_parameter_views_device() const override;

    void copy_parameters_host();

    void copy_parameters_device();

    void allocate_parameters_device();

    void free_parameters_device();

    bool use_combinations = true;

private:

    float* biases_device = nullptr;
    float* weights_device = nullptr;
    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    // Batch Normalization
    float* bn_scale_device = nullptr;
    float* bn_offset_device = nullptr;
    float* bn_running_mean_device = nullptr;
    float* bn_running_variance_device = nullptr;
    cudnnTensorDescriptor_t bn_tensor_descriptor = nullptr;

#endif

private:

    Tensor1 biases;

    Tensor2 weights;

    bool batch_normalization = false;

    Tensor1 scales;
    Tensor1 offsets;

    Tensor1 moving_means;
    Tensor1 moving_standard_deviations;

    type momentum = type(0.9);

    string activation_function = "HyperbolicTangent";

    type dropout_rate = type(0);
};


#ifdef OPENNN_CUDA

struct DenseForwardPropagation<2>Cuda : public LayerForwardPropagationCuda
{
    DenseForwardPropagation<2>Cuda(const Index& = 0, Layer* = nullptr);

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    void free() override;

    float* combinations = nullptr;

    cudnnTensorDescriptor_t output_softmax_tensor_descriptor = nullptr;

    cudnnTensorDescriptor_t biases_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t biases_add_tensor_descriptor = nullptr;

    cudnnDropoutDescriptor_t dropout_descriptor = nullptr;
    void* dropout_states = nullptr;
    size_t dropout_states_size = 0;
    unsigned long long dropout_seed;

    void* dropout_reserve_space = nullptr;
    size_t dropout_reserve_space_size = 0;

    float* bn_saved_mean = nullptr;
    float* bn_saved_inv_variance = nullptr;
};


struct DenseBackPropagation<2>Cuda : public LayerBackPropagationCuda
{
    DenseBackPropagation<2>Cuda(const Index& = 0, Layer* = nullptr);

    vector<ParameterView> get_parameter_delta_views_device() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    void free() override;

    float* bias_deltas_device = nullptr;
    float* weight_deltas_device = nullptr;

    float* ones = nullptr;

    cudnnTensorDescriptor_t deltas_tensor_descriptor = nullptr;

    float* bn_scale_deltas_device = nullptr;
    float* bn_offset_deltas_device = nullptr;
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
