//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensors.h"
#include "random_utilities.h"

namespace opennn
{

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

struct LayerForwardPropagationCuda;
struct LayerBackPropagationCuda;

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

class Layer
{

public:

    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Layer();
    virtual ~Layer();

    const string& get_label() const;

    bool get_display() const;

    const string& get_name() const;

    virtual void set_input_shape(const Shape&);
    virtual void set_output_shape(const Shape&);

    void set_label(const string&);

    void set_display(bool);

    virtual void set_parameters_random();

    virtual void set_parameters_glorot();

    Index get_parameters_number();

    virtual vector<TensorView*> get_parameter_views()
    {
        return vector<TensorView*>();
    }

    //virtual pair

    virtual Shape get_input_shape() const = 0;
    virtual Shape get_output_shape() const = 0;

    Index get_inputs_number() const;

    Index get_outputs_number() const;

    void set_threads_number(const int&);

    // Forward propagation

    virtual void forward_propagate(const vector<TensorView>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   bool) = 0;

    // Back propagation

    virtual void back_propagate(const vector<TensorView>&,
                                const vector<TensorView>&,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>&) const {}

    virtual void back_propagate_lm(const vector<TensorView>&,
                                   const vector<TensorView>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   unique_ptr<LayerBackPropagationLM>&) const {}

    virtual void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                                   Index,
                                                   Tensor2&) const {}

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void to_XML(tinyxml2::XMLPrinter&) const {}

    virtual string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const;

    virtual void print() const {}

    vector<string> get_default_feature_names() const;

    vector<string> get_default_output_names() const;

    bool get_is_trainable() const;

protected:

    unique_ptr<ThreadPool> thread_pool = nullptr;
    unique_ptr<ThreadPoolDevice> device = nullptr;

    string label = "my_layer";

    string name = "layer";

    bool is_trainable = true;

    Tensor2 empty_2;
    Tensor3 empty_3;
    Tensor4 empty_4;

    bool display = true;

    template <int Rank>
    void calculate_activations(const string& activation_function,
                               TensorMapR<Rank> activations,
                               TensorMapR<Rank> activation_derivatives) const
    {
        if (activation_function == "Linear")
            linear(activations, activation_derivatives);
        else if (activation_function == "Sigmoid")
            logistic(activations, activation_derivatives);
        else if (activation_function == "Softmax")
            softmax(activations);
        else if (activation_function == "Competitive")
            throw runtime_error("Competitive 3d not implemented");
        else if (activation_function == "HyperbolicTangent")
            hyperbolic_tangent(activations, activation_derivatives);
        else if (activation_function == "RectifiedLinear")
            rectified_linear(activations, activation_derivatives);
        else if (activation_function == "ScaledExponentialLinear")
            exponential_linear(activations, activation_derivatives);
        else
            throw runtime_error("Unknown activation: " + activation_function);
    }


    template <int Rank>
    FORCE_INLINE void binary(TensorR<Rank>& y, TensorR<Rank>& dy_dx, type threshold) const
    {
        y.device(*device) = (y < threshold).select(type(0), type(1));

        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(0));
    }


    template <int Rank>
    FORCE_INLINE void linear(TensorMapR<Rank>, TensorMapR<Rank> dy_dx) const
    {
        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(1));
    }


    template <int Rank>
    FORCE_INLINE void exponential_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        const type alpha = type(1);

        y.device(*device) = (y > type(0)).select(y, alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*device) = (y > type(0)).select(dy_dx.constant(type(1)), y + alpha);
    }


    template <int Rank>
    FORCE_INLINE void hyperbolic_tangent(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        y.device(*device) = y.tanh();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*device) = (type(1) - y.square()).eval();
    }


    template <int Rank>
    FORCE_INLINE void logistic(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        y.device(*device) = (type(1) + (-y).exp()).inverse();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*device) = (y * (type(1) - y)).eval();
    }


    template <int Rank>
    FORCE_INLINE void rectified_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        y.device(*device) = y.cwiseMax(type(0));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*device) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(0)));
    }


    template <int Rank>
    FORCE_INLINE void leaky_rectified_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx, type slope) const
    {
        y.device(*device) = (y > type(0)).select(y, slope * y);

        if (dy_dx.size() == 0) return;

        dy_dx.device(*device) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(slope)));
    }


    template <int Rank>
    FORCE_INLINE void scaled_exponential_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        const type lambda = type(1.0507);

        const type alpha = type(1.6733);

        y.device(*device) = (y > type(0)).select(lambda * y, lambda * alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*device) = (y > type(0)).select(dy_dx.constant(lambda), y + alpha * lambda);
    }

    void softmax(TensorMap2) const;
    void softmax(TensorMap3) const;
    void softmax(TensorMap4) const;

    void softmax_derivatives_times_tensor(const TensorMap3, TensorMap3, TensorMap1) const;

    void add_gradients(const vector<TensorView>& output_gradient_views) const;

    template <int Rank>
    void normalize_batch(
        TensorMapR<Rank> outputs,
        TensorMapR<Rank> normalized_outputs,
        TensorMap1 batch_means,
        TensorMap1 batch_variances,
        Tensor1 running_means,
        Tensor1 running_variances,
        const TensorMap1 gammas,
        const TensorMap1 betas,
        bool is_training,
        const type momentum = type(0.9),
        const type epsilon = type(1e-5)) const
    {
        const Index neurons = running_means.size();

        array<int, Rank - 1> reduction_axes;
        iota(reduction_axes.begin(), reduction_axes.end(), 0);

        array<Index, Rank> reshape_dimensions;
        reshape_dimensions.fill(1);
        reshape_dimensions.back() = neurons;

        array<Index, Rank> broadcast_dimensions = outputs.dimensions();
        broadcast_dimensions.back() = 1;

        if(is_training)
        {
            batch_means.device(*device) = outputs.mean(reduction_axes);

            normalized_outputs.device(*device) = (outputs - batch_means.reshape(reshape_dimensions).broadcast(broadcast_dimensions));

            batch_variances.device(*device) = (normalized_outputs.square().mean(reduction_axes) + epsilon).sqrt();

            normalized_outputs.device(*device) = normalized_outputs / batch_variances.reshape(reshape_dimensions).broadcast(broadcast_dimensions);

            running_means.device(*device) = running_means * momentum + batch_means * (type(1) - momentum);
            running_variances.device(*device) = running_variances * momentum + batch_variances * (type(1) - momentum);
        }
        else
            normalized_outputs.device(*device) = (outputs - running_means.reshape(reshape_dimensions).broadcast(broadcast_dimensions)) /
                                                 (running_variances.reshape(reshape_dimensions).broadcast(broadcast_dimensions) + epsilon);

        outputs.device(*device) = normalized_outputs * gammas.reshape(reshape_dimensions).broadcast(broadcast_dimensions) +
                                  betas.reshape(reshape_dimensions).broadcast(broadcast_dimensions);
    }


    template <int Rank>
    void dropout(TensorMapR<Rank> tensor, type dropout_rate) const
    {
        const type scaling_factor = type(1) / (type(1) - dropout_rate);

        #pragma omp parallel
        {
            for(Index i = 0; i < tensor.size(); i++)
                tensor(i) = (random_uniform(0, 1) < dropout_rate)
                                ? 0
                                : tensor(i) * scaling_factor;
        }
    }


    template <int Rank>
    void calculate_combinations(const TensorMapR<Rank> inputs,
                                const TensorMap2 weights,
                                const TensorMap1 biases,
                                TensorMapR<Rank> outputs) const
    {
        const Index inputs_size = weights.dimension(0);
        const Index outputs_size = weights.dimension(1);
        const Index total_rows = inputs.size() / inputs_size;

        const Map<const Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            inputs_matrix(inputs.data(), total_rows, inputs_size);

        const Map<const Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            weights_matrix(weights.data(), inputs_size, outputs_size);

        const Map<const Eigen::RowVector<type, Dynamic>>
            biases_vector(biases.data(), outputs_size);

        Map<Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            outputs_matrix(outputs.data(), total_rows, outputs_size);

        outputs_matrix.noalias() = (inputs_matrix * weights_matrix).rowwise() + biases_vector;
    }


    template <int Rank>
    void calculate_gradients(const TensorMapR<Rank> inputs,
                             const TensorMapR<Rank> output_gradients,
                             const TensorMap2 weights,
                             TensorMap2 weight_gradients,
                             TensorMap1 bias_gradients,
                             TensorMapR<Rank> input_gradients,
                             const bool is_first_layer) const
    {
        const Index inputs_size = weights.dimension(0);
        const Index outputs_size = weights.dimension(1);
        const Index total_rows = inputs.size() / inputs_size;

        const Map<const Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            inputs_matrix(inputs.data(), total_rows, inputs_size);

        const Map<const Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            gradients_matrix(output_gradients.data(), total_rows, outputs_size);

        Map<Eigen::Vector<type, Dynamic>>
            bias_gradients_vector(bias_gradients.data(), outputs_size);

        Map<Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            weight_gradients_matrix(weight_gradients.data(), inputs_size, outputs_size);

        bias_gradients_vector.noalias() = gradients_matrix.colwise().sum();

        weight_gradients_matrix.noalias() = inputs_matrix.transpose() * gradients_matrix;

        if(!is_first_layer)
        {
            const Map<const Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
                weights_matrix(weights.data(), inputs_size, outputs_size);

            Map<Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
                input_gradients_matrix(input_gradients.data(), total_rows, inputs_size);

            input_gradients_matrix.noalias() = gradients_matrix * weights_matrix.transpose();
        }
    }

#ifdef OPENNN_CUDA

public:

    void create_cuda();
    void destroy_cuda();

    cudnnHandle_t get_cudnn_handle();

    virtual void forward_propagate(const vector<TensorViewCuda>&,
                                   unique_ptr<LayerForwardPropagationCuda>&,
                                   bool)
    {
        throw runtime_error("CUDA forward propagation not implemented for layer type: " + get_name());
    }

    virtual void back_propagate(const vector<TensorViewCuda>&,
                                const vector<TensorViewCuda>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                unique_ptr<LayerBackPropagationCuda>&) const 
    {
        throw runtime_error("CUDA back propagation not implemented for layer type: " + get_name());
    }

    virtual vector<TensorViewCuda*> get_parameter_views_device()
    {
        return vector<TensorViewCuda*>();
    }

    virtual void free() {}

    virtual void print_parameters_cuda() {}

protected:

    cublasHandle_t cublas_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;

    cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;

#endif

};


struct LayerForwardPropagation
{
    LayerForwardPropagation() {}
    virtual ~LayerForwardPropagation() = default;

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_workspace_views();

    TensorView get_outputs() const;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    TensorView outputs;
};


struct LayerBackPropagation
{
    LayerBackPropagation() {}
    virtual ~LayerBackPropagation() = default;

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_gradient_views();

    vector<TensorView> get_input_gradients() const;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    vector<TensorView> input_gradients;
    vector<Tensor1> input_gradients_memory;
};


struct LayerBackPropagationLM
{
    LayerBackPropagationLM() {}
    virtual ~LayerBackPropagationLM() = default;

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_gradient_views();

    vector<TensorView> get_input_gradients() const;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    vector<TensorView> input_gradients;
    vector<Tensor1> input_gradients_memory;
};


#ifdef OPENNN_CUDA

struct LayerForwardPropagationCuda
{
    LayerForwardPropagationCuda() {}
    virtual ~LayerForwardPropagationCuda() {}

    void set(const Index = 0, Layer* = nullptr);

    virtual void initialize() = 0;

    virtual vector<TensorViewCuda*> get_workspace_views();

    TensorViewCuda get_outputs() const;

    virtual void print() const {}

    virtual void free() {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    TensorViewCuda outputs;
};


struct LayerBackPropagationCuda
{
    LayerBackPropagationCuda() {}

    virtual ~LayerBackPropagationCuda() {}

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorViewCuda*> get_workspace_views()
    {
		return vector<TensorViewCuda*>();
    };

    vector<TensorViewCuda> get_input_gradient_views() const;

    virtual void print() const {}

	virtual void free() {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    vector<TensorCuda> input_gradients;
};

#endif

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
