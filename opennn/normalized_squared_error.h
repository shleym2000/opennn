//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NORMALIZEDSQUAREDERROR_H
#define NORMALIZEDSQUAREDERROR_H

// System includes

#include <cmath>
#include <string>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "data_set.h"

namespace opennn
{

class NormalizedSquaredError : public LossIndex
{

public:

    // Constructors

   explicit NormalizedSquaredError(NeuralNetwork*, DataSet*);

   explicit NormalizedSquaredError();   

   // Get methods

    type get_normalization_coefficient() const;
    type get_selection_normalization_coefficient() const;

   // Set methods

    void set_normalization_coefficient();
    void set_normalization_coefficient(const type&);

    void set_time_series_normalization_coefficient();

    void set_selection_normalization_coefficient();
    void set_selection_normalization_coefficient(const type&);

    void set_default();

    void set_data_set(DataSet* new_data_set) final;

   // Normalization coefficients 

   type calculate_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

   type calculate_time_series_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Back propagation
     
   void calculate_error(const Batch&,
                        const ForwardPropagation&,
                        BackPropagation&) const final;

   void calculate_output_delta(const Batch&,
                               ForwardPropagation&,
                               BackPropagation&) const final;

    // Back propagation LM

   void calculate_error_lm(const Batch&,
                           const ForwardPropagation&,
                           BackPropagationLM&) const final;

   void calculate_output_delta_lm(const Batch&,
                               ForwardPropagation&,
                               BackPropagationLM&) const final;

   void calculate_error_gradient_lm(const Batch&,
                              BackPropagationLM&) const final;

   void calculate_error_hessian_lm(const Batch&,
                                        BackPropagationLM&) const final;

   // Serialization methods

   string get_error_type() const final;
   string get_error_type_text() const final;

   virtual void from_XML(const tinyxml2::XMLDocument&) const;

   void write_XML(tinyxml2::XMLPrinter&) const final;

private:

   type normalization_coefficient = type(NAN);

   type selection_normalization_coefficient = type(NAN);

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/normalized_squared_error_cuda.h"
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
