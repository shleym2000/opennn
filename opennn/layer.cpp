//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <vector>

#include "layer.h"
#include "random_utilities.h"

namespace opennn
{


void LayerForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


void LayerBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


vector<TensorView> LayerBackPropagation::get_input_deltas() const
{
    return input_deltas;
}


#ifdef OPENNN_CUDA

void LayerForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


TensorViewCuda LayerForwardPropagationCuda::get_outputs_device() const
{
    return outputs;
}


vector<TensorViewCuda*> LayerForwardPropagationCuda::get_workspace_views_device()
{
    return { &outputs };
}


void LayerBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if(!new_layer) return;

    batch_size = new_batch_size;
    layer = new_layer;

    initialize();
}


vector<TensorViewCuda> LayerBackPropagationCuda::get_input_deltas_device() const
{
	return input_deltas;
}

#endif


Layer::Layer()
{
    const unsigned int threads_number = thread::hardware_concurrency();

    thread_pool = make_unique<ThreadPool>(threads_number);
    device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);
}


Layer::~Layer() = default;


const bool& Layer::get_display() const
{
    return display;
}


const string& Layer::get_label() const
{
    return label;
}


const string& Layer::get_name() const
{
    return name;
}


void Layer::set_label(const string& new_label)
{
    label = new_label;
}


void Layer::set_display(const bool& new_display)
{
    display = new_display;
}


void Layer::set_parameters_random()
{
    const vector<TensorView*> parameter_views = get_parameter_views();

    for(const auto& view : parameter_views)
    {
        TensorMap1 this_parameters(view->data, view->size());

        set_random_uniform(this_parameters);
    }
}


void Layer::set_parameters_glorot()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const type limit = sqrt(6.0 / (inputs_number + outputs_number));

    const vector<TensorView*> parameter_views = get_parameter_views();

    for(const TensorView* view : parameter_views)
    {
        TensorMap1 this_parameters(view->data, view->size());

        set_random_uniform(this_parameters, -limit, limit);
    }
}


Index Layer::get_parameters_number()
{
    vector<TensorView*> parameter_views = get_parameter_views();

    Index parameters_number = 0;

    for(Index i = 0; i < Index(parameter_views.size()); i++)
        parameters_number += parameter_views[i]->size();

    return parameters_number;
}


void Layer::set_threads_number(const int& new_threads_number)
{
    thread_pool.reset();
    device.reset();

    thread_pool = make_unique<ThreadPool>(new_threads_number);
    device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


string Layer::get_expression(const vector<string> &, const vector<string> &) const
{
    return string();
}


vector<string> Layer::get_default_feature_names() const
{
    const Index inputs_number = get_inputs_number();

    vector<string> feature_names(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        feature_names[i] = "input_" + to_string(i);

    return feature_names;
}


vector<string> Layer::get_default_output_names() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> output_names(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        output_names[i] = "output_" + to_string(i);

    return output_names;
}


bool Layer::get_is_trainable() const
{
    return is_trainable;
}


void Layer::add_deltas(const vector<TensorView> &delta_views) const
{
    TensorMap3 deltas = tensor_map<3>(delta_views[0]);

    for(Index i = 1; i < Index(delta_views.size()); i++)
        deltas.device(*device) += tensor_map<3>(delta_views[i]);
}


Index Layer::get_inputs_number() const
{
    const dimensions input_dimensions = get_input_dimensions();

    return count_elements(input_dimensions);
}


Index Layer::get_outputs_number() const
{
    const dimensions output_dimensions = get_output_dimensions();

    return accumulate(output_dimensions.begin(), output_dimensions.end(), 1, multiplies<Index>());
}


void Layer::forward_propagate(const vector<TensorView>&,
                              unique_ptr<LayerForwardPropagation>&, const bool&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}



void Layer::set_input_dimensions(const dimensions&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}


void Layer::set_output_dimensions(const dimensions&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}


void Layer::softmax(TensorMap2 y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);

    #pragma omp parallel for
    for(Index i = 0; i < rows_number; i++)
    {
        type max_value = -numeric_limits<type>::infinity();
        for(Index j = 0; j < columns_number; j++)
            if(y(i, j) > max_value)
                max_value = y(i, j);

        type sum = 0.0;
        for(Index j = 0; j < columns_number; j++)
        {
            y(i, j) = exp(y(i, j) - max_value);
            sum += y(i, j);
        }

        if(sum > 0.0)
        {
            const type inv_sum = type(1.0) / sum;

            for(Index j = 0; j < columns_number; j++)
                y(i, j) *= inv_sum;
        }
    }
}


void Layer::softmax(TensorMap3 y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);

    #pragma omp parallel for collapse(2)
    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            type max_value = -numeric_limits<type>::infinity();

            for(Index k = 0; k < channels; k++)
                if(y(i, j, k) > max_value)
                    max_value = y(i, j, k);

            type sum = 0.0;
            for(Index k = 0; k < channels; k++)
            {
                y(i, j, k) = exp(y(i, j, k) - max_value);
                sum += y(i, j, k);
            }

            if(sum > 0.0)
            {
                const type inv_sum = type(1.0) / sum;

                for(Index k = 0; k < channels; k++)
                    y(i, j, k) *= inv_sum;
            }
        }
    }
}


void Layer::softmax(TensorMap4 y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);
    const Index blocks_number = y.dimension(3);

    #pragma omp parallel for collapse(3)
    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            for(Index k = 0; k < channels; k++)
            {
                type max_value = -std::numeric_limits<type>::infinity();

                for(Index l = 0; l < blocks_number; l++)
                    if(y(i, j, k, l) > max_value)
                        max_value = y(i, j, k, l);

                type sum = 0.0;
                for(Index l = 0; l < blocks_number; l++)
                {
                    y(i, j, k, l) = exp(y(i, j, k, l) - max_value);
                    sum += y(i, j, k, l);
                }

                if(sum > 0.0)
                {
                    const type inv_sum = type(1.0) / sum;

                    for(Index l = 0; l < blocks_number; l++)
                        y(i, j, k, l) *= inv_sum;
                }
            }
        }
    }
}


void Layer::softmax_derivatives_times_tensor(const TensorMap3 softmax,
                                             TensorMap3 result,
                                             TensorMap1 aux_rows) const
{
    const Index rows = softmax.dimension(0);
    const Index columns = softmax.dimension(1);
    const Index depth = softmax.dimension(2);


    type* softmax_data = (type*)softmax.data();
    type* result_data = result.data();

    type* softmax_vector_data = nullptr;
    type* result_vector_data = nullptr;

    Tensor<type, 0> sum;

    for(Index i = 0; i < depth; i++)
    {
        for(Index j = 0; j < columns; j++)
        {
            softmax_vector_data = softmax_data + rows * (i * columns + j);
            result_vector_data = result_data + rows * (i * columns + j);

            const TensorMap1 softmax_vector(softmax_vector_data, rows);
            const TensorMap1 tensor_vector(result_vector_data, rows);

            TensorMap1 result_vector(result_vector_data, rows);

            aux_rows.device(*device) = softmax_vector * tensor_vector;

            sum.device(*device) = aux_rows.sum();

            result_vector.device(*device) = aux_rows - softmax_vector * sum(0);
        }
    }
}


#ifdef OPENNN_CUDA

void Layer::create_cuda()
{
    cublasCreate(&cublas_handle);
    cudnnCreate(&cudnn_handle);

    // Multiplication

    cudnnCreateOpTensorDescriptor(&operator_multiplication_descriptor);

    cudnnSetOpTensorDescriptor(operator_multiplication_descriptor,
                               CUDNN_OP_TENSOR_MUL,
                               CUDNN_DATA_FLOAT,
                               CUDNN_NOT_PROPAGATE_NAN);

    // Sum

    cudnnCreateOpTensorDescriptor(&operator_sum_descriptor);

    cudnnSetOpTensorDescriptor(operator_sum_descriptor,
                               CUDNN_OP_TENSOR_ADD,
                               CUDNN_DATA_FLOAT,
                               CUDNN_NOT_PROPAGATE_NAN);
}


void Layer::destroy_cuda()
{
    cublasDestroy(cublas_handle);
    cudnnDestroy(cudnn_handle);

    cudnnDestroyOpTensorDescriptor(operator_multiplication_descriptor);
    cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
}


cudnnHandle_t Layer::get_cudnn_handle()
{
    return cudnn_handle;
}

#endif

TensorView LayerForwardPropagation::get_outputs() const
{
    return outputs;
}

vector<TensorView*> LayerForwardPropagation::get_workspace_views()
{
    return {&outputs};
}

vector<TensorView> LayerBackPropagationLM::get_input_deltas() const
{
    return input_deltas;
}

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
