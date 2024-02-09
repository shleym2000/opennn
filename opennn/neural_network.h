//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <errno.h>

// OpenNN includes

#include "config.h"
#include "data_set_batch.h"
#include "layer.h"
#include "addition_layer.h"
#include "perceptron_layer.h"
#include "perceptron_layer_3d.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_4d.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "probabilistic_layer.h"
#include "probabilistic_layer_3d.h"
#include "convolutional_layer.h"
#include "flatten_layer.h"
#include "non_max_suppression_layer.h"
#include "embedding_layer.h"
#include "multihead_attention_layer.h"

#include "pooling_layer.h"
#include "long_short_term_memory_layer.h"
#include "recurrent_layer.h"
#include "text_analytics.h"

namespace opennn
{

struct ForwardPropagation;
struct NeuralNetworkBackPropagation;
struct NeuralNetworkBackPropagationLM;

/// This class represents the concept of neural network in the OpenNN library.
///
/// This neural network spans a function space for the problem at hand.

class NeuralNetwork
{

public:

   enum class ModelType{AutoAssociation,
                        Approximation,
                        Classification,
                        Forecasting,
                        ImageClassification,
                        TextClassification};

   // Constructors

   explicit NeuralNetwork();

   explicit NeuralNetwork(const NeuralNetwork::ModelType&, const Tensor<Index, 1>&);

   explicit NeuralNetwork(const NeuralNetwork::ModelType&, const initializer_list<Index>&);

   explicit NeuralNetwork(const Tensor<Index, 1>&, const Index&, const Tensor<Index, 1>&, const Index&);

   explicit NeuralNetwork(const string&);

   explicit NeuralNetwork(const tinyxml2::XMLDocument&);

   explicit NeuralNetwork(const Tensor<Layer*, 1>&);

   // Destructor

   virtual ~NeuralNetwork();

   // APPENDING LAYERS

   void delete_layers();

   void add_layer(Layer*);

   bool check_layer_type(const Layer::Type);

   // Get methods

   bool has_scaling_layer() const;
   bool has_long_short_term_memory_layer() const;
   bool has_recurrent_layer() const;
   bool has_unscaling_layer() const;
   bool has_bounding_layer() const;
   bool has_probabilistic_layer() const;
   bool has_convolutional_layer() const;
   bool has_flatten_layer() const;
   bool is_empty() const;

   const Tensor<string, 1>& get_inputs_names() const;
   string get_input_name(const Index&) const;
   Index get_input_index(const string&) const;

   ModelType get_model_type() const;
   string get_model_type_string() const;

   const Tensor<string, 1>& get_outputs_names() const;
   string get_output_name(const Index&) const;
   Index get_output_index(const string&) const;

   Tensor<Layer*, 1> get_layers_pointers() const;
   Layer* get_layer_pointer(const Index&) const;
   Tensor<Layer*, 1> get_trainable_layers_pointers() const;
   Tensor<Index, 1> get_trainable_layers_indices() const;   

   Index get_layer_index(const string&) const;

   Tensor<Tensor<Index, 1>, 1> get_layers_inputs_indices() const;

   ScalingLayer2D* get_scaling_layer_2d_pointer() const;
   ScalingLayer4D* get_scaling_layer_4d_pointer() const;
   UnscalingLayer* get_unscaling_layer_pointer() const;
   BoundingLayer* get_bounding_layer_pointer() const;
   FlattenLayer* get_flatten_layer_pointer() const;
   //ConvolutionalLayer* get_convolutional_layer_pointer() const;
   PoolingLayer* get_pooling_layer_pointer() const;
   ProbabilisticLayer* get_probabilistic_layer_pointer() const;
   LongShortTermMemoryLayer* get_long_short_term_memory_layer_pointer() const;
   RecurrentLayer* get_recurrent_layer_pointer() const;

   Layer* get_last_trainable_layer_pointer() const;
   PerceptronLayer* get_first_perceptron_layer_pointer() const;

   Index get_batch_samples_number() const;

   const bool& get_display() const;

   // Set methods

   void set();

   void set(const NeuralNetwork::ModelType&, const Tensor<Index, 1>&);
   void set(const NeuralNetwork::ModelType&, const initializer_list<Index>&);
   void set(const Tensor<Index, 1>&, const Index&, const Tensor<Index, 1>&, const Index&);

   void set(const string&);

   void set_layers_pointers(Tensor<Layer*, 1>&);

   void set_layers_inputs_indices(const Tensor<Tensor<Index, 1>, 1>&);
   void set_layer_inputs_indices(const Index&, const Tensor<Index, 1>&);

   void set_layer_inputs_indices(const string&, const Tensor<string, 1>&);
   void set_layer_inputs_indices(const string&, const initializer_list<string>&);
   void set_layer_inputs_indices(const string&, const string&);

   void set_model_type(const ModelType&);
   void set_model_type_string(const string&);
   void set_inputs_names(const Tensor<string, 1>&);
   void set_outputs_names(const Tensor<string, 1>&);

   void set_inputs_number(const Index&);
   void set_inputs_number(const Tensor<bool, 1>&);

   virtual void set_default();

   void set_threads_number(const int&);

   void set_scaling_layer_2d(ScalingLayer2D&);
   void set_scaling_layer_4d(ScalingLayer4D&);

   void set_display(const bool&);

   // Layers

   Index get_layers_number() const;
   Tensor<Index, 1> get_layers_neurons_numbers() const;

   Index get_trainable_layers_number() const;
   Index get_first_trainable_layer_index() const;
   Index get_last_trainable_layer_index() const;

   Index get_perceptron_layers_number() const;
   Index get_probabilistic_layers_number() const;
   Index get_flatten_layers_number() const;
   Index get_convolutional_layers_number() const;
   Index get_pooling_layers_number() const;
   Index get_long_short_term_memory_layers_number() const;
   Index get_recurrent_layers_number() const;

   bool is_starting_layer(const Index&) const;

   // Architecture

   Index get_inputs_number() const;
   Index get_outputs_number() const;

   Tensor<Index, 1> get_trainable_layers_neurons_numbers() const;
   Tensor<Index, 1> get_trainable_layers_inputs_numbers() const;

   Tensor<Index, 1> get_architecture() const;

   // Parameters

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   Tensor<Index, 1> get_layers_parameters_numbers() const;
   Tensor<Index, 1> get_trainable_layers_parameters_numbers() const;

   void set_parameters(Tensor<type, 1>&) const;

   // Parameters initialization methods

   void set_parameters_constant(const type&) const;

   void set_parameters_random() const;

   // Parameters

   type calculate_parameters_norm() const;

   // Output

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   Tensor<type, 2> calculate_scaled_outputs(const Tensor<type, 2>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 4>&);

   Tensor<type, 2> calculate_directional_inputs(const Index&, const Tensor<type, 1>&, const type&, const type&, const Index& = 101) const;

   // Serialization methods

   Tensor<string, 2> get_information() const;
   Tensor<string, 2> get_perceptron_layers_information() const;
   Tensor<string, 2> get_probabilistic_layer_information() const;

   virtual void from_XML(const tinyxml2::XMLDocument&);
   void inputs_from_XML(const tinyxml2::XMLDocument&);
   void layers_from_XML(const tinyxml2::XMLDocument&);
   void outputs_from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void print() const;
   void save(const string&) const;
   void save_parameters(const string&) const;

   virtual void load(const string&) final;
   void load_parameters_binary(const string&);

   Tensor<string, 1> get_layers_names() const;

   // Expression methods

   string write_expression() const;

   string write_expression_python() const;
   string write_expression_c() const;
   string write_expression_api() const;
   string write_expression_javascript() const;

   void save_expression_c(const string&) const;
   void save_expression_python(const string&) const;
   void save_expression_api(const string&) const;
   void save_expression_javascript(const string&) const;
   void save_outputs(Tensor<type, 2>&, const string&);

   /// Calculate forward propagation in neural network

   void forward_propagate(const pair<type*, dimensions>&, ForwardPropagation&, const bool& = false) const;

   void forward_propagate(const pair<type*, dimensions>&, Tensor<type, 1>&, ForwardPropagation&) const;

protected:

   string name = "neural_network";

   NeuralNetwork::ModelType model_type;

   /// Names of inputs

   Tensor<string, 1> inputs_names;

   /// Names of ouputs

   Tensor<string, 1> outputs_names;

   /// Layers

   Tensor<Layer*, 1> layers_pointers;

   Tensor<Tensor<Index, 1>, 1> layers_inputs_indices;

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/neural_network_cuda.h"
#endif

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
