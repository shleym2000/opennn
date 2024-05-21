//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER3D_H
#define PERCEPTRONLAYER3D_H

// System includes

#include <cstdlib>
#include <iostream>
#include <string>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "layer_back_propagation.h"
#include "probabilistic_layer_3d.h"
#include "multihead_attention_layer.h"


namespace opennn
{

struct PerceptronLayer3DForwardPropagation;
struct PerceptronLayer3DBackPropagation;

#ifdef OPENNN_CUDA
struct PerceptronLayer3DForwardPropagationCuda;
struct PerceptronLayer3DBackPropagationCuda;
#endif


/// This class represents a layer of perceptrons.

/// PerceptronLayer is a single-layer network with a hard-limit transfer function.
/// This network is often trained with the perceptron learning rule.
///
/// Layers of perceptrons will be used to construct multilayer perceptrons, such as an approximation problems .


class PerceptronLayer3D : public Layer
{

public:

    /// Enumeration of the available activation functions for the perceptron neuron model.

    enum class ActivationFunction{HyperbolicTangent,
                                  Linear,
                                  RectifiedLinear};

   // Constructors

   explicit PerceptronLayer3D();

   explicit PerceptronLayer3D(const Index&,
                              const Index&,
                              const Index&,
                              const ActivationFunction& = PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const final;
   Index get_inputs_depth() const;
   Index get_neurons_number() const final;

   dimensions get_outputs_dimensions() const final;

   // Parameters

   const Tensor<type, 1>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_synaptic_weights(const Tensor<type, 1>&) const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;
   Index get_parameters_number() const final;
   type get_dropout_rate() const;
   Tensor<type, 1> get_parameters() const final;

   // Activation functions

   const PerceptronLayer3D::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&,
            const Index&,
            const Index&,
            const PerceptronLayer3D::ActivationFunction& = PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

   void set_default();
   void set_name(const string&);

   // Architecture

   void set_inputs_number(const Index&) final;
   void set_inputs_depth(const Index&);
   void set_neurons_number(const Index&) final;

   // Parameters

   void set_biases(const Tensor<type, 1>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0) final;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);
   void set_dropout_rate(const type&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);

   void set_parameters_constant(const type&) final;
   void set_parameters_random() final;
   void set_parameters_glorot();

   // Forward propagation

   void calculate_combinations(const Tensor<type, 3>&,
                               const Tensor<type, 1>&,
                               const Tensor<type, 2>&,
                               Tensor<type, 3>&) const;

   void dropout(Tensor<type, 3>&) const;

   void calculate_activations(const Tensor<type, 3>&,
                              Tensor<type, 3>&) const;

   void calculate_activations_derivatives(const Tensor<type, 3>&,
                                          Tensor<type, 3>&,
                                          Tensor<type, 3>&) const;

   void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                          LayerForwardPropagation*,
                          const bool&) final;

   // Gradient methods

   void back_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                                 const Tensor<pair<type*, dimensions>, 1>&,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const final;

   void add_deltas(const Tensor<pair<type*, dimensions>, 1>&) const;

   void insert_gradient(LayerBackPropagation*,
                        const Index&,
                        Tensor<type, 1>&) const final;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;
   void write_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_cuda.h"
    #endif

protected:

   // MEMBERS

   Index inputs_number;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's transfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   /// This matrix contains conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function;

   type dropout_rate = type(0);

   /// Display messages to screen.

   bool display = true;


};


struct PerceptronLayer3DForwardPropagation : LayerForwardPropagation
{
    // Default constructor

     explicit PerceptronLayer3DForwardPropagation() : LayerForwardPropagation()
     {
     }


     explicit PerceptronLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
         : LayerForwardPropagation()
     {        
        set(new_batch_samples_number, new_layer);
     }


    virtual ~PerceptronLayer3DForwardPropagation()
    {
    }
    
    
    pair<type*, dimensions> get_outputs_pair() const final;


    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


     void print() const
     {
         cout << "Outputs:" << endl;
         cout << outputs << endl;

         cout << "Activations derivatives:" << endl;
         cout << activations_derivatives << endl;
     }

     Tensor<type, 3> outputs;

     Tensor<type, 3> activations_derivatives;
};


struct PerceptronLayer3DBackPropagation : LayerBackPropagation
{
    // Default constructor

    explicit PerceptronLayer3DBackPropagation() : LayerBackPropagation()
    {

    }


    explicit PerceptronLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    virtual ~PerceptronLayer3DBackPropagation()
    {
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const
    {
        cout << "Biases derivatives:" << endl;
        cout << biases_derivatives << endl;

        cout << "Synaptic weights derivatives:" << endl;
        cout << synaptic_weights_derivatives << endl;
    }

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;

    Tensor<type, 3> error_combinations_derivatives;
    Tensor<type, 3> input_derivatives;
};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_back_propagation_cuda.h"
#endif


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
