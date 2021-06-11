//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "mean_squared_error.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a mean squared error term not associated to any
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

MeanSquaredError::MeanSquaredError() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a mean squared error term object associated to a
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// Destructor.

MeanSquaredError::~MeanSquaredError()
{
}


/// \brief MeanSquaredError::calculate_error
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void MeanSquaredError::calculate_error(const DataSetBatch& batch,
                     const NeuralNetworkForwardPropagation&,
                     LossIndexBackPropagation& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    const Index batch_samples_number = batch.inputs_2d.dimension(0);

    const type coefficient = static_cast<type>(batch_samples_number);

    sum_squared_error.device(*thread_pool_device) = back_propagation.errors.contract(back_propagation.errors, SSE);

    back_propagation.error = sum_squared_error(0)/coefficient;
}


void MeanSquaredError::calculate_error_lm(const DataSetBatch& batch,
                     const NeuralNetworkForwardPropagation&,
                     LossIndexBackPropagationLM& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    const Index batch_samples_number = batch.inputs_2d.dimension(0);

//    sum_squared_error.device(*thread_pool_device) = (back_propagation.squared_errors*back_propagation.squared_errors).sum();
    sum_squared_error.device(*thread_pool_device) = back_propagation.squared_errors.sum();

    const type coefficient = static_cast<type>(batch_samples_number);

    back_propagation.error = sum_squared_error(0)/coefficient;
}


void MeanSquaredError::calculate_output_delta(const DataSetBatch& batch,
                                              NeuralNetworkForwardPropagation&,
                                              LossIndexBackPropagation& back_propagation) const
{
     #ifdef OPENNN_DEBUG
     check();
     #endif

     const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

     LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(trainable_layers_number-1);

     const Index batch_samples_number = batch.inputs_2d.dimension(0);

     const type coefficient = static_cast<type>(2.0)/static_cast<type>(batch_samples_number);

     switch(output_layer_back_propagation->layer_pointer->get_type())
     {
     case Layer::Perceptron:
     {
         PerceptronLayerBackPropagation* perceptron_layer_back_propagation
         = static_cast<PerceptronLayerBackPropagation*>(output_layer_back_propagation);

         perceptron_layer_back_propagation->delta.device(*thread_pool_device) = coefficient*back_propagation.errors;
     }
         break;

     case Layer::Probabilistic:
     {
         ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation
         = static_cast<ProbabilisticLayerBackPropagation*>(output_layer_back_propagation);

         probabilistic_layer_back_propagation->delta.device(*thread_pool_device) = coefficient*back_propagation.errors;
     }
         break;

     case Layer::Recurrent:
     {
         RecurrentLayerBackPropagation* recurrent_layer_back_propagation
         = static_cast<RecurrentLayerBackPropagation*>(output_layer_back_propagation);

         recurrent_layer_back_propagation->delta.device(*thread_pool_device) = coefficient*back_propagation.errors;
     }
         break;

     case Layer::LongShortTermMemory:
     {
         LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation
         = static_cast<LongShortTermMemoryLayerBackPropagation*>(output_layer_back_propagation);

         long_short_term_memory_layer_back_propagation->delta.device(*thread_pool_device) = coefficient*back_propagation.errors;
     }
         break;

     default: break;
     }
}


void MeanSquaredError::calculate_output_delta_lm(const DataSetBatch&,
                                                 NeuralNetworkForwardPropagation&,
                                                 LossIndexBackPropagationLM& loss_index_back_propagation) const
{
#ifdef OPENNN_DEBUG
    check();
#endif

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerBackPropagationLM* output_layer_back_propagation = loss_index_back_propagation.neural_network.layers(trainable_layers_number-1);

    Layer* output_layer_pointer = output_layer_back_propagation->layer_pointer;

    switch(output_layer_pointer->get_type())
    {
    case Layer::Perceptron:
    {
        PerceptronLayerBackPropagationLM* perceptron_layer_back_propagation
                = static_cast<PerceptronLayerBackPropagationLM*>(output_layer_back_propagation);

        memcpy(perceptron_layer_back_propagation->delta.data(),
               loss_index_back_propagation.errors.data(),
               static_cast<size_t>(loss_index_back_propagation.errors.size())*sizeof(type));

        divide_columns(perceptron_layer_back_propagation->delta, loss_index_back_propagation.squared_errors);
    }
        break;

    case Layer::Probabilistic:
    {
        ProbabilisticLayerBackPropagationLM* probabilistic_layer_back_propagation
                = static_cast<ProbabilisticLayerBackPropagationLM*>(output_layer_back_propagation);

        memcpy(probabilistic_layer_back_propagation->delta.data(),
               loss_index_back_propagation.errors.data(),
               static_cast<size_t>(loss_index_back_propagation.errors.size())*sizeof(type));

        divide_columns(probabilistic_layer_back_propagation->delta, loss_index_back_propagation.squared_errors);
    }
        break;

    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MeanSquaredError class.\n"
               << "Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n";

        throw logic_error(buffer.str());
    }
    }
}


void MeanSquaredError::calculate_error_gradient_lm(const DataSetBatch& batch,
                                             LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index batch_samples_number = batch.get_samples_number();

    const type coefficient = static_cast<type>(2)/static_cast<type>(batch_samples_number);

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device)
            = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors, AT_B);

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device)
            = coefficient*loss_index_back_propagation_lm.gradient;
}


void MeanSquaredError::calculate_error_hessian_lm(const DataSetBatch& batch,
                                                       LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
     #ifdef OPENNN_DEBUG
     check();
     #endif

     const Index batch_samples_number = batch.inputs_2d.dimension(0);

     const type coefficient = (static_cast<type>(2.0)/static_cast<type>(batch_samples_number));

     loss_index_back_propagation_lm.hessian.device(*thread_pool_device)
             = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors_jacobian, AT_B);

     loss_index_back_propagation_lm.hessian.device(*thread_pool_device)
             = coefficient*loss_index_back_propagation_lm.hessian;
}


/// Returns a string with the name of the mean squared error loss type, "MEAN_SQUARED_ERROR".

string MeanSquaredError::get_error_type() const
{
    return "MEAN_SQUARED_ERROR";
}


/// Returns a string with the name of the mean squared error loss type in text format.

string MeanSquaredError::get_error_type_text() const
{
    return "Mean squared error";
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void MeanSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("MeanSquaredError");

    file_stream.CloseElement();
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
