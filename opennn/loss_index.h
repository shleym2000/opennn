//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S   H E A D E R                         
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LOSSINDEX_H
#define LOSSINDEX_H

// System includes

#include <string>

// OpenNN includes

#include "config.h"
#include "data_set.h"
#include "batch.h"
#include "neural_network.h"
#include "neural_network_back_propagation_lm.h"

namespace opennn
{

struct BackPropagation;
struct BackPropagationLM;

#ifdef OPENNN_CUDA
struct BackPropagationCuda;
#endif


class LossIndex
{

public:

   // Constructors

   explicit LossIndex();

   explicit LossIndex(NeuralNetwork*, DataSet*);

   // Destructor

   virtual ~LossIndex();

   // Methods

   enum class RegularizationMethod{L1, L2, NoRegularization};

   inline NeuralNetwork* get_neural_network() const 
   {
        #ifdef OPENNN_DEBUG

        if(!neural_network)
             throw runtime_error("Neural network pointer is nullptr.\n");

        #endif

      return neural_network;
   }


   inline DataSet* get_data_set() const 
   {
        #ifdef OPENNN_DEBUG

        if(!data_set)
             throw runtime_error("DataSet pointer is nullptr.\n");

        #endif

      return data_set;
   }

   const type& get_regularization_weight() const;

   const bool& get_display() const;

   bool has_neural_network() const;

   bool has_data_set() const;

   // Get

   RegularizationMethod get_regularization_method() const;

   // Set

   void set();
   void set(NeuralNetwork*);
   void set(DataSet*);
   void set(NeuralNetwork*, DataSet*);

   void set(const LossIndex&);

   void set_threads_number(const int&);

   void set_neural_network(NeuralNetwork*);

   virtual void set_data_set(DataSet*);

   void set_default();

   void set_regularization_method(const RegularizationMethod&);
   void set_regularization_method(const string&);
   void set_regularization_weight(const type&);

   void set_display(const bool&);

   virtual void set_normalization_coefficient() {}

   bool has_selection() const;

   Index find_input_index(const vector<Index>&, const Index&) const;

   // Numerical differentiation

   type calculate_eta() const;
   type calculate_h(const type&) const;

   Tensor<type, 1> calculate_numerical_gradient();

   Tensor<type, 2> calculate_numerical_jacobian();

   Tensor<type, 1> calculate_numerical_inputs_derivatives();

   // Back propagation

   virtual void calculate_error(const Batch&,
                                const ForwardPropagation&,
                                BackPropagation&) const = 0;

   void add_regularization(BackPropagation&) const;

   virtual void calculate_output_delta(const Batch&,
                                       ForwardPropagation&,
                                       BackPropagation&) const = 0;

   void calculate_layers_error_gradient(const Batch& ,
                                        ForwardPropagation& ,
                                        BackPropagation&) const;

   void assemble_layers_error_gradient(BackPropagation&) const;

   void back_propagate(const Batch&,
                       ForwardPropagation&,
                       BackPropagation&) const;

   // Back propagation LM

   void calculate_errors_lm(const Batch&,
                            const ForwardPropagation&,
                            BackPropagationLM&) const; // general

   virtual void calculate_squared_errors_lm(const Batch&,
                                            const ForwardPropagation&,
                                            BackPropagationLM&) const;

   virtual void calculate_error_lm(const Batch&,
                                   const ForwardPropagation&,
                                   BackPropagationLM&) const {}

   virtual void calculate_output_delta_lm(const Batch&,
                                          ForwardPropagation&,
                                          BackPropagationLM&) const {}

   void calculate_layers_squared_errors_jacobian_lm(const Batch&,
                                                    ForwardPropagation&,
                                                    BackPropagationLM&) const;

   virtual void calculate_error_gradient_lm(const Batch&,
                                      BackPropagationLM&) const;

   virtual void calculate_error_hessian_lm(const Batch&,
                                           BackPropagationLM&) const {}

   void back_propagate_lm(const Batch&,
                          ForwardPropagation&,
                          BackPropagationLM&) const;

   // Regularization

   type calculate_regularization(const Tensor<type, 1>&) const;

   void calculate_regularization_gradient(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_regularization_hessian(Tensor<type, 1>&, Tensor<type, 2>&) const;

   // Serialization

   void from_XML(const tinyxml2::XMLDocument&);

   virtual void to_XML(tinyxml2::XMLPrinter&) const;

   void regularization_from_XML(const tinyxml2::XMLDocument&);
   void write_regularization_XML(tinyxml2::XMLPrinter&) const;

   virtual string get_loss_method() const;
   virtual string get_error_type_text() const;

   string write_regularization_method() const;

   // Checking

   void check() const;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/neural_network_cuda.h"
        #include "../../opennn_cuda/opennn_cuda/loss_index_cuda.h"

        #include "../../opennn_cuda/opennn_cuda/mean_squared_error_cuda.h"
        #include "../../opennn_cuda/opennn_cuda/cross_entropy_error_cuda.h"
    #endif

protected:

   ThreadPool* thread_pool = nullptr;

   ThreadPoolDevice* thread_pool_device = nullptr;

   NeuralNetwork* neural_network = nullptr;

   DataSet* data_set = nullptr;

   RegularizationMethod regularization_method = RegularizationMethod::L2;

   type regularization_weight = type(0.01);

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
   const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

   const Eigen::array<IndexPair<Index>, 2> SSE = {IndexPair<Index>(0, 0), IndexPair<Index>(1, 1)};

   const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/loss_index_back_propagation_cuda.h"
#endif

struct BackPropagationLM
{
    BackPropagationLM();

    explicit BackPropagationLM(const Index&, LossIndex*);

    void set(const Index&, LossIndex*);

    void print() const;
    
    void set_layers_outputs_indices(const vector<vector<Index>>&);

    pair<type*, dimensions> get_output_deltas_pair() const;

    vector<vector<pair<type*, dimensions>>> get_layers_deltas(const Index last_trainable_layer_index, const Index first_trainable_layer_index) const
{
    vector<vector<pair<type*, dimensions>>> layers_deltas(neural_network.get_layers().size());

    const auto& layers_outputs_indices = this->layers_outputs_indices; // Output indices for each layer
    const auto& layers_inputs_indices = neural_network.get_neural_network()->get_layers_input_indices();

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; --i)
    {
        if (i == last_trainable_layer_index)
        {
            // For the last layer, use the output deltas
            layers_deltas[i].push_back(get_output_deltas_pair());
        }
        else
        {
            // For other layers, determine the deltas based on the input derivatives of the next layers
            layers_deltas[i].resize(layers_outputs_indices[i].size());

            for (Index j = 0; j < layers_outputs_indices[i].size(); ++j)
            {
                Index output_index = layers_outputs_indices[i][j];
                Index input_index = loss_index->find_input_index(layers_inputs_indices[output_index], i);

                LayerBackPropagationLM* layer_back_propagation = neural_network.get_layers()[output_index];

                // Access the input derivatives using the appropriate index
                layers_deltas[i][j] = layer_back_propagation->get_inputs_derivatives_pair()[input_index];
            }
        }
    }

    return layers_deltas;
}


    Index batch_samples_number = 0;

    Tensor<Tensor<Index, 1>, 1> layers_outputs_indices;

    Tensor<type, 1> output_deltas;
    dimensions output_deltas_dimensions;

    LossIndex* loss_index = nullptr;

    type error = type(0);
    type regularization = type(0);
    type loss = type(0);

    Tensor<type, 1> parameters;

    NeuralNetworkBackPropagationLM neural_network;

    Tensor<type, 2> errors;
    Tensor<type, 1> squared_errors;
    Tensor<type, 2> squared_errors_jacobian;

    Tensor<type, 1> gradient;
    Tensor<type, 2> hessian;

    Tensor<type, 1> regularization_gradient;
    Tensor<type, 2> regularization_hessian;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
