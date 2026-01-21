//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"
#include "random_utilities.h"

namespace opennn
{

template<int Rank> class Dense;

template<int Rank>
struct DenseForwardPropagation final : LayerForwardPropagation
{
    DenseForwardPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    virtual ~DenseForwardPropagation() = default;

    void initialize() override
    {
        const auto* dense_layer = static_cast<const Dense<Rank>*>(layer);
        const dimensions output_dims = dense_layer->get_output_dimensions();

        dimensions full_output_dims = {batch_size};
        full_output_dims.insert(full_output_dims.end(), output_dims.begin(), output_dims.end());

        outputs.dims = full_output_dims;
        activation_derivatives.dims = full_output_dims;

        if (dense_layer->get_batch_normalization())
        {
            const Index outputs_number = dense_layer->get_outputs_number();
            means.dims = {outputs_number};
            standard_deviations.dims = {outputs_number};
            normalized_outputs.dims = full_output_dims;
        }
    }

    vector<TensorView*> get_tensor_views() override
    {
        vector<TensorView*> views = { &outputs, &activation_derivatives };

        if (means.size() > 0)
        {
            views.push_back(&means);
            views.push_back(&standard_deviations);
            views.push_back(&normalized_outputs);
        }

        return views;
    }


    void print() const override
    {
        cout << "Outputs:" << endl
             << outputs.data << endl
             << "Activation derivatives:" << endl
             << activation_derivatives.data << endl;
    }

    TensorView means;
    TensorView standard_deviations;
    TensorView normalized_outputs;
    TensorView activation_derivatives;
};


template<int Rank>
struct DenseBackPropagation final : LayerBackPropagation
{
    DenseBackPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    virtual ~DenseBackPropagation() = default;


    void initialize() override
    {
        const auto* dense_layer = static_cast<const Dense<Rank>*>(layer);

        const Index outputs_number = layer->get_outputs_number();
        const Index inputs_number = layer->get_input_dimensions()[0];

        bias_deltas.dims = {outputs_number};
        weight_deltas.dims = {inputs_number, outputs_number};

        const dimensions input_dims = dense_layer->get_input_dimensions();
        dimensions full_input_dims = {batch_size};
        full_input_dims.insert(full_input_dims.end(), input_dims.begin(), input_dims.end());

        input_deltas_tensor.resize(full_input_dims);
        input_deltas.resize(1);
        input_deltas[0] = TensorView(input_deltas_tensor.data(), full_input_dims);

        if (dense_layer->get_batch_normalization())
        {
            bn_scale_deltas.dims = {outputs_number};
            bn_offset_deltas.dims = {outputs_number};
        }
    }


    vector<TensorView*> get_tensor_views() override
    {
        vector<TensorView*> views = {&bias_deltas, &weight_deltas};

        const auto* dense_layer = static_cast<const Dense<Rank>*>(layer);

        if (dense_layer->get_batch_normalization())
        {
            views.push_back(&bn_scale_deltas);
            views.push_back(&bn_offset_deltas);
        }

        return views;
    }


    void print() const override
    {
        cout << "Bias deltas:" << endl
             << bias_deltas.data << endl
             << "Weight deltas:" << endl
             << weight_deltas.data << endl;
    }

    TensorView bias_deltas;
    TensorView weight_deltas;

    TensorView bn_scale_deltas;
    TensorView bn_offset_deltas;

    Tensor<type, Rank> input_deltas_tensor;
};


template<int Rank>
struct DenseBackPropagationLM final : LayerBackPropagationLM
{
    DenseBackPropagationLM(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    void set(const Index& new_samples_number = 0, Layer* new_layer = nullptr) override
    {
        layer = new_layer;
        batch_size = new_samples_number;

        const Index parameters_number = layer->get_parameters_number();
        const dimensions layer_input_dims = layer->get_input_dimensions();

        dimensions input_dims_vec = {batch_size};
        input_dims_vec.insert(input_dims_vec.end(), layer_input_dims.begin(), layer_input_dims.end());

        input_deltas.dims = input_dims_vec;

        squared_errors_Jacobian.dims = {batch_size, parameters_number};
    }

    vector<TensorView> get_input_deltas() const override
    {
        return { input_deltas };
    }

    vector<TensorView*> get_tensor_views() override
    {
        return { &input_deltas, &squared_errors_Jacobian };
    }

    void print() const override
    {
        cout << "Squared errors Jacobian: " << endl;
        squared_errors_Jacobian.print();
        cout << "Input derivatives: " << endl;
        input_deltas.print();
    }

    TensorView input_deltas;
    TensorView squared_errors_Jacobian;
};


#ifdef OPENNN_CUDA

template<int Rank>
struct DenseForwardPropagationCuda : public LayerForwardPropagationCuda
{
    DenseForwardPropagationCuda(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }


    void initialize() override
    {
        auto* dense_layer = static_cast<Dense<Rank>*>(this->layer);
        const Index outputs_number = dense_layer->get_output_dimensions().back();

        Index total_rows = batch_size;
        if constexpr (Rank == 3)
            total_rows *= dense_layer->get_input_dimensions()[0];

        cudnnCreateTensorDescriptor(&biases_add_tensor_descriptor);
        cudnnSetTensor4dDescriptor(biases_add_tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, (int)outputs_number, (int)total_rows, 1);

        CHECK_CUDA(cudaMalloc(&outputs.data, total_rows * outputs_number * sizeof(float)));
        if (dense_layer->use_combinations)
            CHECK_CUDA(cudaMalloc(&combinations, total_rows * outputs_number * sizeof(float)));

        cudnnCreateTensorDescriptor(&output_softmax_tensor_descriptor);
        cudnnSetTensor4dDescriptor(output_softmax_tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, (int)outputs_number, (int)total_rows, 1);

        cudnnCreateTensorDescriptor(&outputs.descriptor);
        cudnnSetTensor4dDescriptor(outputs.descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)total_rows, (int)outputs_number, 1, 1);

        if (dense_layer->get_dropout_rate() > 0)
        {
            cudnnCreateDropoutDescriptor(&dropout_descriptor);
            cudnnDropoutGetStatesSize(dense_layer->get_cudnn_handle(), &dropout_states_size);
            CHECK_CUDA(cudaMalloc(&dropout_states, dropout_states_size));
            cudnnSetDropoutDescriptor(dropout_descriptor, dense_layer->get_cudnn_handle(), (float)dense_layer->get_dropout_rate(), dropout_states, dropout_states_size, dropout_seed);
            cudnnDropoutGetReserveSpaceSize(outputs.descriptor, &dropout_reserve_space_size);
            CHECK_CUDA(cudaMalloc(&dropout_reserve_space, dropout_reserve_space_size));
        }

        if (dense_layer->get_batch_normalization())
        {
            CHECK_CUDA(cudaMalloc(&bn_saved_mean, outputs_number * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&bn_saved_inv_variance, outputs_number * sizeof(float)));
        }
    }


    void free() override
    {
        if (combinations) cudaFree(combinations);
        if (outputs.data) cudaFree(outputs.data);
        if (dropout_states) cudaFree(dropout_states);
        if (dropout_reserve_space) cudaFree(dropout_reserve_space);
        if (bn_saved_mean) cudaFree(bn_saved_mean);
        if (bn_saved_inv_variance) cudaFree(bn_saved_inv_variance);

        cudnnDestroyTensorDescriptor(output_softmax_tensor_descriptor);
        cudnnDestroyTensorDescriptor(outputs.descriptor);
        cudnnDestroyTensorDescriptor(biases_add_tensor_descriptor);
        if (dropout_descriptor) cudnnDestroyDropoutDescriptor(dropout_descriptor);
    }

    float* combinations = nullptr;

    cudnnTensorDescriptor_t output_softmax_tensor_descriptor = nullptr;
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

template<int Rank>
struct DenseBackPropagationCuda : public LayerBackPropagationCuda
{
    DenseBackPropagationCuda(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerBackPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }


    void initialize() override
    {
        auto* dense_layer = static_cast<Dense<Rank>*>(this->layer);
        const Index outputs_number = dense_layer->get_output_dimensions().back();
        const Index inputs_number = dense_layer->get_input_dimensions().back();

        Index total_rows = batch_size;
        if constexpr (Rank == 3)
            total_rows *= dense_layer->get_input_dimensions()[0];

        CHECK_CUDA(cudaMalloc(&ones, total_rows * sizeof(float)));
        vector<float> ones_host(total_rows, 1.0f);
        CHECK_CUDA(cudaMemcpy(ones, ones_host.data(), total_rows * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&bias_deltas_device.data, outputs_number * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&weight_deltas_device.data, inputs_number * outputs_number * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&input_deltas[0].data, total_rows * inputs_number * sizeof(float)));

        cudnnCreateTensorDescriptor(&deltas_tensor_descriptor);
        cudnnSetTensor4dDescriptor(deltas_tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)total_rows, (int)outputs_number, 1, 1);

        if (dense_layer->get_batch_normalization())
        {
            CHECK_CUDA(cudaMalloc(&bn_scale_deltas_device.data, outputs_number * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&bn_offset_deltas_device.data, outputs_number * sizeof(float)));
        }
    }


    vector<TensorViewCuda*> get_tensor_views_device() override
    {
        /*
        auto* dense_layer = static_cast<Dense<Rank>*>(this->layer);

        vector<TensorViewCuda*> views = {
            {bias_deltas_device, dense_layer->get_output_dimensions().back()},
            {weight_deltas_device, dense_layer->get_input_dimensions().back() * dense_layer->get_output_dimensions().back()}
        };

        if (dense_layer->get_batch_normalization())
        {
            views.push_back({bn_scale_deltas_device, dense_layer->get_output_dimensions().back()});
            views.push_back({bn_offset_deltas_device, dense_layer->get_output_dimensions().back()});
        }
    */
        vector<TensorViewCuda*> views;
        return views;
    }

    void free() override
    {
        if (bias_deltas_device.data) cudaFree(bias_deltas_device.data);
        if (weight_deltas_device.data) cudaFree(weight_deltas_device.data);
        if (input_deltas[0].data) cudaFree(input_deltas[0].data);
        if (ones) cudaFree(ones);
        if (bn_scale_deltas_device.data) cudaFree(bn_scale_deltas_device.data);
        if (bn_offset_deltas_device.data) cudaFree(bn_offset_deltas_device.data);
        cudnnDestroyTensorDescriptor(deltas_tensor_descriptor);
    }

    TensorViewCuda bias_deltas_device;
    TensorViewCuda weight_deltas_device;

    float* ones = nullptr;

    TensorViewCuda bn_scale_deltas_device;
    TensorViewCuda bn_offset_deltas_device;
    cudnnTensorDescriptor_t deltas_tensor_descriptor = nullptr;
};

#endif


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


    Dense(const Index& input_sequence_length,
          const Index& embedding_dimension,
          const Index& feed_forward_dimension,
          const string& new_activation_function,
          const string& new_label)
    {
        set({input_sequence_length, embedding_dimension},
            {feed_forward_dimension},
            new_activation_function,
            false,
            new_label);
    }


    dimensions get_input_dimensions() const override
    {
        return { weights.dims[0] };
    }


    dimensions get_output_dimensions() const override
    {
        return { biases.size() };
    }


    vector<TensorView*> get_parameter_views() override
    {
        vector<TensorView*> parameter_views = {&biases, &weights};

        if (batch_normalization)
        {
            parameter_views.push_back(&scales);
            parameter_views.push_back(&offsets);
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


    const string& get_activation_function() const
    {
        return activation_function;
    }


    void set(const dimensions& new_input_dimensions = {},
             const dimensions& new_output_dimensions = {},
             const string& new_activation_function = "HyperbolicTangent",
             const bool& new_batch_normalization = false,
             const string& new_label = "dense_layer")
    {
        if (new_input_dimensions.size() != Rank - 1)
            throw runtime_error("Input dimensions size must be " + to_string(Rank - 1));

        if (new_output_dimensions.size() != 1)
            throw runtime_error("Output dimensions size is not 1");

        biases.dims = { new_output_dimensions[0] };
        weights.dims = { new_input_dimensions[0], new_output_dimensions[0] };

        set_activation_function(new_activation_function);

        set_batch_normalization(new_batch_normalization);

        const Index outputs_number = get_outputs_number();

        if (batch_normalization)
        {
            scales.dims = {outputs_number};
            offsets.dims = {outputs_number};
            moving_means.resize(outputs_number);
            moving_standard_deviations.resize(outputs_number);
        }

        set_label(new_label);

        name = "Dense" + to_string(Rank) + "d";

#ifdef OPENNN_CUDA

        cudnnCreateTensorDescriptor(&biases_tensor_descriptor);
        cudnnSetTensor4dDescriptor(biases_tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outputs_number, 1, 1);

        biases_device.descriptor = biases_tensor_descriptor;
        weights_device.descriptor = nullptr;

        if (batch_normalization)
        {
            cudnnCreateTensorDescriptor(&bn_tensor_descriptor);

            cudnnSetTensor4dDescriptor(bn_tensor_descriptor,
                                       CUDNN_TENSOR_NCHW,
                                       CUDNN_DATA_FLOAT,
                                       1, outputs_number, 1, 1);

            bn_offset_device.descriptor = bn_tensor_descriptor;
            bn_scale_device.descriptor = bn_tensor_descriptor;
        }

#endif
    }


    void set_parameters_glorot() override
    {
        const type limit = sqrt(6.0 / (get_inputs_number() + get_outputs_number()));

        if(biases.size() > 0)
        {
            TensorMap1 biases_map(biases.data, biases.size());
            biases_map.setZero();
        }

        if(weights.size() > 0)
        {
            TensorMap1 weights_map(weights.data, weights.size());
            set_random_uniform(weights_map, -limit, limit);
        }

        if(batch_normalization)
        {
            if(scales.size() > 0)
            {
                TensorMap1 scales_map(scales.data, scales.size());
                scales_map.setConstant(1.0);
            }
            if(offsets.size() > 0)
            {
                TensorMap1 offsets_map(offsets.data, offsets.size());
                offsets_map.setZero();
            }
        }
    }


    void set_parameters_random() override
    {
        if(biases.size() > 0)
        {
            TensorMap1 biases_map(biases.data, biases.size());
            biases_map.setZero();
        }

        if(weights.size() > 0)
        {
            TensorMap1 weights_map(weights.data, weights.size());
            set_random_uniform(weights_map);
        }

        if (batch_normalization)
        {
            if(scales.size() > 0)
                TensorMap1(scales.data, scales.size()).setConstant(1.0);

            if(offsets.size() > 0)
                TensorMap1(offsets.data, offsets.size()).setZero();
        }
    }


    void set_input_dimensions(const dimensions& new_input_dimensions) override
    {
        const Index inputs_number = new_input_dimensions[0];
        const Index outputs_number = get_outputs_number();

        biases.dims = { outputs_number };
        weights.dims = { inputs_number, outputs_number };
    }


    void set_output_dimensions(const dimensions& new_output_dimensions) override
    {
        const Index inputs_number = get_inputs_number();
        const Index neurons_number = new_output_dimensions[0];

        biases.dims = { neurons_number };
        weights.dims = { inputs_number, neurons_number };
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


    void set_dropout_rate(const type& new_dropout_rate)
    {
        if (new_dropout_rate < type(0) || new_dropout_rate >= type(1))
            throw runtime_error("Dropout rate must be in [0,1).");

        dropout_rate = new_dropout_rate;
    }

/*
    void normalization(Tensor1& means, Tensor1& standard_deviations, const Tensor2& inputs, Tensor2& outputs) const
    {
        const array<Index, 2> rows({outputs.dimension(0), 1});

        const array<int, 1> axis_x({0});

        means.device(*device) = outputs.mean(axis_x);

        standard_deviations.device(*device)
            = (outputs - means.broadcast(rows)).square().mean(axis_x).sqrt();

        outputs = inputs;// -means.broadcast(array<Index, 2>({ outputs.dimension(0), 1 }));
            //shifts.broadcast(rows);
            //+ (outputs - means.broadcast(rows))*scales.broadcast(rows)/standard_deviations.broadcast(rows);
    }
*/

    void set_batch_normalization(const bool& new_batch_normalization)
    {
        batch_normalization = new_batch_normalization;
    }


    void apply_batch_normalization_backward(TensorMap2& deltas,
                                            unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                            unique_ptr<LayerBackPropagation>& layer_back_propagation) const
    {
        const DenseForwardPropagation<2>* dense2d_forward_propagation =
            static_cast<const DenseForwardPropagation<2>*>(layer_forward_propagation.get());

        const Index batch_size = dense2d_forward_propagation->batch_size;

        const TensorMap2 normalized_outputs = tensor_map<2>(dense2d_forward_propagation->normalized_outputs);
        const TensorMap1 standard_deviations = tensor_map<1>(dense2d_forward_propagation->standard_deviations);

        DenseBackPropagation<2>* dense2d_back_propagation =
            static_cast<DenseBackPropagation<2>*>(layer_back_propagation.get());

        TensorMap1 bn_scale_deltas = tensor_map<1>(dense2d_back_propagation->bn_scale_deltas);
        TensorMap1 bn_offset_deltas = tensor_map<1>(dense2d_back_propagation->bn_offset_deltas);

        const array<int, 1> reduction_axes = { 0 };
        const array<Index, 2> reshape_dims = { 1, get_outputs_number() };
        const array<Index, 2> broadcast_dims = { batch_size, 1 };

        bn_offset_deltas.device(*device) = deltas.sum(reduction_axes);
        bn_scale_deltas.device(*device) = (deltas * normalized_outputs).sum(reduction_axes);

        const auto inv_m = type(1) / batch_size;
/*
        deltas.device(*device) =
            ((deltas * type(batch_size))
             - bn_offset_deltas.reshape(reshape_dims).broadcast(broadcast_dims)
             - normalized_outputs *
                   bn_scale_deltas.reshape(reshape_dims).broadcast(broadcast_dims)
             ) * inv_m
            / standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims)
            * scales.reshape(reshape_dims).broadcast(broadcast_dims);
*/
    }


    void forward_propagate(const vector<TensorView>& input_views,
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           const bool& is_training) override
    {
        auto* dense_forward_propagation = static_cast<DenseForwardPropagation<Rank>*>(layer_forward_propagation.get());

        auto outputs = tensor_map<Rank>(dense_forward_propagation->outputs);

        calculate_combinations<Rank>(tensor_map<Rank>(input_views[0]), tensor_map<2>(weights), tensor_map<1>(biases), outputs);

        if(batch_normalization)
        {
            auto normalized_outputs = tensor_map<Rank>(dense_forward_propagation->normalized_outputs);

            normalize_batch<Rank>(
                outputs,
                normalized_outputs,
                tensor_map<1>(dense_forward_propagation->means),
                tensor_map<1>(dense_forward_propagation->standard_deviations),
                moving_means,
                moving_standard_deviations,
                tensor_map<1>(scales),
                tensor_map<1>(offsets),
                is_training,
                momentum);
        }

        if(is_training)
        {
            auto activation_derivatives = tensor_map<Rank>(dense_forward_propagation->activation_derivatives);
            calculate_activations<Rank>(activation_function, outputs, activation_derivatives);
        }
        else
        {
            if constexpr(Rank == 2)
                calculate_activations<Rank>(activation_function, outputs, TensorMap2(empty_2.data(), empty_2.dimensions()));
            else if constexpr(Rank == 3)
                calculate_activations<Rank>(activation_function, outputs, TensorMap3(empty_3.data(), empty_3.dimensions()));
        }

        if(is_training && dropout_rate > type(0))
            dropout<Rank>(outputs, dropout_rate);
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

        const TensorMap2 activation_derivatives = tensor_map<2>(dense2d_layer_forward_propagation->activation_derivatives);

        // Back propagation

        TensorMap2 input_deltas = tensor_map<2>(back_propagation->input_deltas[0]);

        DenseBackPropagation<2>* dense2d_back_propagation =
            static_cast<DenseBackPropagation<2>*>(back_propagation.get());

        TensorMap2 weight_deltas = tensor_map<2>(dense2d_back_propagation->weight_deltas);

        TensorMap1 bias_deltas = tensor_map<1>(dense2d_back_propagation->bias_deltas);

        const bool& is_first_layer = dense2d_back_propagation->is_first_layer;

        if(activation_function != "Softmax")
            deltas.device(*device) = deltas * activation_derivatives;

        if (batch_normalization)
            apply_batch_normalization_backward(deltas, forward_propagation, back_propagation);

        bias_deltas.device(*device) = deltas.sum(array_1(0));

        weight_deltas.device(*device) = inputs.contract(deltas, axes(0,0));

        if (!is_first_layer)
            input_deltas.device(*device) = deltas.contract(tensor_map<2>(weights), axes(1,1));
    }


    void back_propagate_lm(const vector<TensorView>& input_views,
                           const vector<TensorView>& delta_views,
                           unique_ptr<LayerForwardPropagation>& forward_propagation,
                           unique_ptr<LayerBackPropagationLM>& back_propagation) const override
    {
        const auto inputs = tensor_map<Rank>(input_views[0]);
        auto deltas = tensor_map<Rank>(delta_views[0]);

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();
        const Index biases_number = biases.size();
        const Index samples_number = inputs.dimension(0);

        const DenseForwardPropagation<Rank>* dense2d_layer_forward_propagation =
            static_cast<DenseForwardPropagation<Rank>*>(forward_propagation.get());

        const auto activation_derivatives
            = tensor_map<Rank>(dense2d_layer_forward_propagation->activation_derivatives);

        DenseBackPropagationLM<Rank>* dense_lm =
            static_cast<DenseBackPropagationLM<Rank>*>(back_propagation.get());

        TensorMap2 squared_errors_Jacobian = tensor_map<2>(dense_lm->squared_errors_Jacobian);

        auto input_deltas = tensor_map<Rank>(dense_lm->input_deltas);

        const bool& is_first_layer = dense_lm->is_first_layer;

        if(activation_function != "Softmax")
            deltas.device(*device) = deltas * activation_derivatives;

        if constexpr(Rank == 2)
            squared_errors_Jacobian.slice(array<Index, 2>{0, 0}, array<Index, 2>{samples_number, biases_number})
                .device(*device) = deltas;
        else
            squared_errors_Jacobian.slice(array<Index, 2>{0, 0}, array<Index, 2>{samples_number, biases_number})
                .device(*device) = deltas.sum(array<Index, Rank-2>{1});

        for(Index j = 0; j < outputs_number; j++)
        {
            const auto delta_j = deltas.chip(j, Rank - 1);

            for(Index i = 0; i < inputs_number; i++)
            {
                const auto input_i = inputs.chip(i, Rank - 1);
                const auto derivative = delta_j * input_i;

                const Index weight_column_index = biases_number + (i * outputs_number) + j;

                if constexpr(Rank == 2)
                    squared_errors_Jacobian.chip(weight_column_index, 1).device(*device) = derivative;
                else
                    squared_errors_Jacobian.chip(weight_column_index, 1).device(*device) = derivative.sum(array<Index, Rank-2>{1});
            }
        }

        if(!is_first_layer)
            input_deltas.device(*device) = deltas.contract(tensor_map<2>(weights), axes(Rank - 1, 1));
    }


    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                           const Index& start_column_index,
                                           Tensor2& global_jacobian) const override
    {
        DenseBackPropagationLM<Rank>* dense_lm =
            static_cast<DenseBackPropagationLM<Rank>*>(back_propagation.get());

        const Index batch_size = dense_lm->batch_size;
        constexpr Index ALIGNMENT = 16;
        constexpr Index MASK = ~(ALIGNMENT - 1);

        Index global_offset = start_column_index;
        Index local_offset = 0;

        if(biases.size() > 0)
        {
            const Index size = biases.size();

            global_jacobian.slice(array<Index, 2>{0, global_offset}, array<Index, 2>{batch_size, size})
                .device(*device) =
                tensor_map<2>(dense_lm->squared_errors_Jacobian)
                    .slice(array<Index, 2>{0, local_offset}, array<Index, 2>{batch_size, size});

            local_offset += size;
            global_offset += (size + ALIGNMENT - 1) & MASK;
        }

        if(weights.size() > 0)
        {
            const Index size = weights.size();

            global_jacobian.slice(array<Index, 2>{0, global_offset}, array<Index, 2>{batch_size, size})
                .device(*device) =
                tensor_map<2>(dense_lm->squared_errors_Jacobian)
                    .slice(array<Index, 2>{0, local_offset}, array<Index, 2>{batch_size, size});

            local_offset += size;
            global_offset += (size + ALIGNMENT - 1) & MASK;
        }

        if(batch_normalization)
        {
            if(scales.size() > 0)
            {
                const Index size = scales.size();

                global_jacobian.slice(array<Index, 2>{0, global_offset}, array<Index, 2>{batch_size, size})
                    .device(*device) = tensor_map<2>(dense_lm->squared_errors_Jacobian)
                          .slice(array<Index, 2>{0, local_offset}, array<Index, 2>{batch_size, size});

                local_offset += size;
                global_offset += (size + ALIGNMENT - 1) & MASK;
            }

            if(offsets.size() > 0)
            {
                const Index size = offsets.size();

                global_jacobian.slice(array<Index, 2>{0, global_offset}, array<Index, 2>{batch_size, size})
                    .device(*device) = tensor_map<2>(dense_lm->squared_errors_Jacobian)
                          .slice(array<Index, 2>{0, local_offset}, array<Index, 2>{batch_size, size});

                local_offset += size;
                global_offset += (size + ALIGNMENT - 1) & MASK;
            }
        }
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
/*
        for(Index j = 0; j < outputs_number; j++)
        {
            const TensorMap1 weights_column = tensor_map(weights, j);

            buffer << output_names[j] << " = " << activation_function << "( " << biases(j) << " + ";

            for(Index i = 0; i < inputs_number - 1; i++)
                buffer << "(" << weights_column(i) << "*" << feature_names[i] << ") + ";

            buffer << "(" << weights_column(inputs_number - 1) << "*" << feature_names[inputs_number - 1] << ") );\n";
        }
*/
        return buffer.str();
    }


    void print() const override
    {
/*
        cout << "Dense layer" << endl
             << "Input dimensions: " << get_input_dimensions()[0] << endl
             << "Output dimensions: " << get_output_dimensions()[0] << endl
             << "Biases dimensions: " << biases.dimensions() << endl
             << "Weights dimensions: " << weights.dimensions() << endl;

        cout << "Activation function:" << endl;
        cout << activation_function << endl;
*/
    }


    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* dense2d_layer_element = document.FirstChildElement(name.c_str());

        if(!dense2d_layer_element)
            throw runtime_error(name + " element is nullptr.\n");

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
/*
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
*/
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement(name.c_str());

        add_xml_element(printer, "Label", label);
        add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
        add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
        add_xml_element(printer, "Activation", activation_function);
        add_xml_element(printer, "BatchNormalization", batch_normalization ? "true" : "false");
/*
        if (batch_normalization)
        {
            add_xml_element(printer, "Scales", tensor_to_string<type, 1>(scales));
            add_xml_element(printer, "Offsets", tensor_to_string<type, 1>(offsets));
            add_xml_element(printer, "MovingMeans", tensor_to_string<type, 1>(moving_means));
            add_xml_element(printer, "MovingStandardDeviations", tensor_to_string<type, 1>(moving_standard_deviations));
        }
*/
        printer.CloseElement();
    }


#ifdef OPENNN_CUDA

public:

    vector<TensorViewCuda*> get_parameter_views_device() override
    {
        vector<TensorViewCuda*> views_device = { &biases_device, &weights_device };

        if (batch_normalization)
        {
            views_device.push_back(&bn_scale_device);
            views_device.push_back(&bn_offset_device);
        }

        return views_device;
    }

    void allocate_parameters_device()
    {

        if (batch_normalization)
        {
            CHECK_CUDA(cudaMalloc(&bn_running_mean_device, outputs_number * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&bn_running_variance_device, outputs_number * sizeof(float)));
        }
    }


    void copy_parameters_device()
    {
        if (batch_normalization)
        {
            CHECK_CUDA(cudaMemcpy(bn_running_mean_device, moving_means.data(), moving_means.size() * sizeof(type), cudaMemcpyHostToDevice));
            Tensor1 moving_variances = moving_standard_deviations.square();
            CHECK_CUDA(cudaMemcpy(bn_running_variance_device, moving_variances.data(), moving_variances.size() * sizeof(type), cudaMemcpyHostToDevice));
        }
    }


    void copy_parameters_host()
    {
        if (batch_normalization)
        {
            CHECK_CUDA(cudaMemcpy(moving_means.data(), bn_running_mean_device, moving_means.size() * sizeof(type), cudaMemcpyDeviceToHost));
            Tensor1 moving_variances(moving_standard_deviations.size());
            CHECK_CUDA(cudaMemcpy(moving_variances.data(), bn_running_variance_device, moving_variances.size() * sizeof(type), cudaMemcpyDeviceToHost));
            moving_standard_deviations = moving_variances.sqrt();
        }
    }


    void forward_propagate_cuda(const vector<TensorViewCuda>& inputs_device,
                                unique_ptr<LayerForwardPropagationCuda>& fp_cuda,
                                const bool& is_training)
    {
        // Dense layer

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        const Index batch_size = fp_cuda->batch_size;

        TensorViewCuda& outputs = fp_cuda->outputs;

        //type* outputs = fp_cuda->outputs.data;
        //const cudnnTensorDescriptor_t output_tensor_descriptor = fp_cuda->outputs.descriptor;

        // Forward propagation

        auto* dense_layer_forward_propagation_cuda = static_cast<DenseForwardPropagationCuda<Rank>*>(fp_cuda.get());

        type* combinations = dense_layer_forward_propagation_cuda->combinations;

        const cudnnTensorDescriptor_t output_softmax_tensor_descriptor = dense_layer_forward_propagation_cuda->output_softmax_tensor_descriptor;

        const cudnnTensorDescriptor_t& biases_add_tensor_descriptor = dense_layer_forward_propagation_cuda->biases_add_tensor_descriptor;
        const cudnnTensorDescriptor_t& biases_tensor_descriptor = biases_device.descriptor;

        type* outputs_buffer = use_combinations ? combinations : outputs.data;

        // Combinations

        cublasSgemm(cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    batch_size, outputs_number, inputs_number,
                    &alpha,
                    inputs_device[0].data,
                    batch_size,
                    weights_device.data,
                    inputs_number,
                    &beta,
                    outputs_buffer,
                    batch_size);

        cudnnStatus_t status = cudnnAddTensor(cudnn_handle,
                                              &alpha,
                                              biases_tensor_descriptor,
                                              biases_device.data,
                                              &beta_add,
                                              biases_add_tensor_descriptor,
                                              outputs_buffer);

        if (status != CUDNN_STATUS_SUCCESS)
            cerr << "Dense CUDA: cudnnAddTensor failed. Error: " << cudnnGetErrorString(status) << endl;

        // Batch Normalization

        if (batch_normalization)
        {
            cudnnStatus_t bn_status;
            constexpr type epsilon = numeric_limits<type>::epsilon();

            if (is_training)
            {
                bn_status = cudnnBatchNormalizationForwardTraining(
                    cudnn_handle,
                    CUDNN_BATCHNORM_PER_ACTIVATION,
                    &alpha, &beta_add,
                    outputs.descriptor,
                    outputs_buffer,
                    outputs.descriptor,
                    outputs_buffer,
                    bn_tensor_descriptor,
                    bn_scale_device.data,
                    bn_offset_device.data,
                    momentum,
                    bn_running_mean_device,
                    bn_running_variance_device,
                    epsilon,
                    dense_layer_forward_propagation_cuda->bn_saved_mean,
                    dense_layer_forward_propagation_cuda->bn_saved_inv_variance);

                if (bn_status != CUDNN_STATUS_SUCCESS)
                    cout << "cudnnBatchNormalizationForwardTraining failed: " << cudnnGetErrorString(bn_status) << endl;
            }
            else
            {
                bn_status = cudnnBatchNormalizationForwardInference(
                    cudnn_handle,
                    CUDNN_BATCHNORM_PER_ACTIVATION,
                    &alpha, &beta_add,
                    outputs.descriptor,
                    outputs_buffer,
                    outputs.descriptor,
                    outputs_buffer,
                    bn_tensor_descriptor,
                    bn_scale_device.data,
                    bn_offset_device.data,
                    bn_running_mean_device,
                    bn_running_variance_device,
                    epsilon);

                if (bn_status != CUDNN_STATUS_SUCCESS)
                    cout << "cudnnBatchNormalizationForwardInference failed: " << cudnnGetErrorString(bn_status) << endl;
            }
        }

        // Activations

        if (activation_function == "Linear")
            cudaMemcpy(outputs.data, combinations, batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
        else if (activation_function == "Softmax")
            cudnnSoftmaxForward(cudnn_handle,
                                CUDNN_SOFTMAX_ACCURATE,
                                CUDNN_SOFTMAX_MODE_CHANNEL,
                                &alpha,
                                output_softmax_tensor_descriptor,
                                combinations,
                                &beta,
                                output_softmax_tensor_descriptor,
                                outputs.data);
        else
            cudnnActivationForward(cudnn_handle,
                                   activation_descriptor,
                                   &alpha,
                                   outputs.descriptor,
                                   outputs_buffer,
                                   &beta,
                                   outputs.descriptor,
                                   outputs.data);

        // Droput

        if (is_training && activation_function != "Softmax" && get_dropout_rate() > type(0))
        {
            status = cudnnDropoutForward(cudnn_handle,
                                         dense_layer_forward_propagation_cuda->dropout_descriptor,
                                         outputs.descriptor,
                                         outputs.data,
                                         outputs.descriptor,
                                         outputs.data,
                                         dense_layer_forward_propagation_cuda->dropout_reserve_space,
                                         dense_layer_forward_propagation_cuda->dropout_reserve_space_size);

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnDropoutForward failed: " << cudnnGetErrorString(status) << endl;
        }
    }


    void back_propagate_cuda(const vector<TensorViewCuda>& inputs_device,
                             const vector<TensorViewCuda>& deltas_device,
                             unique_ptr<LayerForwardPropagationCuda>& fp_cuda,
                             unique_ptr<LayerBackPropagationCuda>& bp_cuda) const
    {
        // Dense layer

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        // Forward propagation

        const Index batch_size = fp_cuda->batch_size;

        const TensorViewCuda& outputs_view = fp_cuda->outputs;

        auto* dense_layer_forward_propagation_cuda = static_cast<DenseForwardPropagationCuda<Rank>*>(fp_cuda.get());

        Dense* dense_layer = static_cast<Dense*>(dense_layer_forward_propagation_cuda->layer);


        type* combinations = dense_layer_forward_propagation_cuda->combinations;

        // Back propagation

        float* input_deltas = bp_cuda->input_deltas[0].data;

        auto* dense_layer_back_propagation = static_cast<DenseBackPropagationCuda<Rank>*>(bp_cuda.get());

        float* ones = dense_layer_back_propagation->ones;

        float* bias_deltas = dense_layer_back_propagation->bias_deltas_device.data;
        float* weight_deltas = dense_layer_back_propagation->weight_deltas_device.data;

        const cudnnTensorDescriptor_t deltas_tensor_descriptor = dense_layer_back_propagation->deltas_tensor_descriptor;

        // Dropout

        if (get_dropout_rate() > type(0) && activation_function != "Softmax")
        {
            const cudnnStatus_t status = cudnnDropoutBackward(cudnn_handle,
                                                        dense_layer_forward_propagation_cuda->dropout_descriptor,
                                                        deltas_tensor_descriptor,
                                                        deltas_device[0].data,
                                                        deltas_tensor_descriptor,
                                                        deltas_device[0].data,
                                                        dense_layer_forward_propagation_cuda->dropout_reserve_space,
                                                        dense_layer_forward_propagation_cuda->dropout_reserve_space_size);

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnDropoutBackward failed: " << cudnnGetErrorString(status) << endl;
        }

        // Error combinations derivatives

        if (dense_layer->get_activation_function() != "Linear" && dense_layer->get_activation_function() != "Softmax")
        {
            if (use_combinations)
            {
                const cudnnStatus_t status = cudnnActivationBackward(cudnn_handle,
                                                               activation_descriptor,
                                                               &alpha,
                                                               deltas_tensor_descriptor,
                                                               outputs_view.data,
                                                               deltas_tensor_descriptor,
                                                               deltas_device[0].data,
                                                               deltas_tensor_descriptor,
                                                               combinations,
                                                               &beta,
                                                               deltas_tensor_descriptor,
                                                               deltas_device[0].data);

                if (status != CUDNN_STATUS_SUCCESS)
                    cout << "cudnnActivationBackward failed: " << cudnnGetErrorString(status) << endl;
            }
            else
            {
                const cudnnStatus_t status = cudnnActivationBackward(cudnn_handle,
                                                               activation_descriptor,
                                                               &alpha,
                                                               deltas_tensor_descriptor,
                                                               outputs_view.data,
                                                               deltas_tensor_descriptor,
                                                               deltas_device[0].data,
                                                               deltas_tensor_descriptor,
                                                               outputs_view.data,
                                                               &beta,
                                                               deltas_tensor_descriptor,
                                                               deltas_device[0].data);

                if (status != CUDNN_STATUS_SUCCESS)
                    cout << "cudnnActivationBackward failed: " << cudnnGetErrorString(status) << endl;
            }
        }

        // Batch Normalization
        constexpr type epsilon = numeric_limits<type>::epsilon();

        if (batch_normalization)
        {
            const cudnnStatus_t bn_status = cudnnBatchNormalizationBackward(
                cudnn_handle,
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha, &beta,
                &alpha, &beta,
                dense_layer_forward_propagation_cuda->outputs.descriptor,
                use_combinations ? combinations : outputs_view.data,
                deltas_tensor_descriptor,
                deltas_device[0].data,
                deltas_tensor_descriptor,
                deltas_device[0].data,
                bn_tensor_descriptor,
                bn_scale_device.data,
                dense_layer_back_propagation->bn_scale_deltas_device.data,
                dense_layer_back_propagation->bn_offset_deltas_device.data,
                epsilon,
                dense_layer_forward_propagation_cuda->bn_saved_mean,
                dense_layer_forward_propagation_cuda->bn_saved_inv_variance);

            if (bn_status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnBatchNormalizationBackward failed: " << cudnnGetErrorString(bn_status) << endl;
        }

        // Bias derivatives

        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    outputs_number,
                    1,
                    batch_size,
                    &alpha,
                    deltas_device[0].data,
                    batch_size,
                    ones,
                    batch_size,
                    &beta,
                    bias_deltas,
                    outputs_number);

        // Weight derivatives

        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    inputs_number,
                    outputs_number,
                    batch_size,
                    &alpha,
                    inputs_device[0].data,
                    batch_size,
                    deltas_device[0].data,
                    batch_size,
                    &beta,
                    weight_deltas,
                    inputs_number);

        // Input derivatives

        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    batch_size,
                    inputs_number,
                    outputs_number,
                    &alpha,
                    deltas_device[0].data,
                    batch_size,
                    weights_device.data,
                    inputs_number,
                    &beta,
                    input_deltas,
                    batch_size);
    }


    void free_parameters_device()
    {
        cudaFree(biases_device.data);
        cudaFree(weights_device.data);

        biases_device.data = nullptr;
        weights_device.data = nullptr;

        cudnnDestroyTensorDescriptor(biases_tensor_descriptor);
        biases_tensor_descriptor = nullptr;

        if (batch_normalization)
        {
            cudaFree(bn_scale_device.data);
            cudaFree(bn_offset_device.data);
            cudaFree(bn_running_mean_device);
            cudaFree(bn_running_variance_device);

            bn_scale_device.data = nullptr;
            bn_offset_device.data = nullptr;
            bn_running_mean_device = nullptr;
            bn_running_variance_device = nullptr;

            cudnnDestroyTensorDescriptor(bn_tensor_descriptor);
            bn_tensor_descriptor = nullptr;
        }
    }

    bool use_combinations = true;

private:

    TensorViewCuda biases_device;
    cudnnTensorDescriptor_t biases_tensor_descriptor = nullptr;

    TensorViewCuda weights_device;

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    // Batch Normalization
    TensorViewCuda bn_scale_device;
    TensorViewCuda bn_offset_device;
    cudnnTensorDescriptor_t bn_tensor_descriptor = nullptr;
    float* bn_running_mean_device = nullptr;
    float* bn_running_variance_device = nullptr;

#endif

private:

    Index inputs_number;
    Index outputs_number;

    TensorView biases;
    TensorView weights;

    TensorView scales;
    TensorView offsets;

    Tensor1 moving_means;
    Tensor1 moving_standard_deviations;

    bool batch_normalization = false;

    type momentum = type(0.9);

    string activation_function = "HyperbolicTangent";

    type dropout_rate = type(0);
};


void reference_dense_layer();

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
