//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"

using type = float;
using namespace std;
using namespace Eigen;

namespace opennn
{

struct Correlation
{
    enum class Method{Pearson, Spearman};

    enum class Form{Linear, Logistic, Logarithmic, Exponential, Power};

    Correlation() {}

    void set_perfect();

    string write_type() const;

    void print() const;

    type a = type(NAN);
    type b = type(NAN);
    type r = type(NAN);

    type lower_confidence = type(NAN);
    type upper_confidence = type(NAN);

    Method method = Method::Pearson;
    Form form = Form::Linear;
};


Correlation linear_correlation(const ThreadPoolDevice*, const Tensor1&, const Tensor1&);

Correlation logarithmic_correlation(const ThreadPoolDevice*, const Tensor1&, const Tensor1&);

Correlation exponential_correlation(const ThreadPoolDevice*, const Tensor1&, const Tensor1&);

Correlation power_correlation(const ThreadPoolDevice*, const Tensor1&, const Tensor1&);

Correlation logistic_correlation_vector_vector(const ThreadPoolDevice*, const Tensor1&, const Tensor1&);

Correlation logistic_correlation_vector_matrix(const ThreadPoolDevice*, const Tensor1&, const Tensor2&);

Correlation logistic_correlation_matrix_vector(const ThreadPoolDevice*, const Tensor2&, const Tensor1&);

Correlation logistic_correlation_matrix_matrix(const ThreadPoolDevice*, const Tensor2&, const Tensor2&);

Correlation correlation(const ThreadPoolDevice*, const Tensor2&, const Tensor2&);

// Spearman correlation

Correlation linear_correlation_spearman(const ThreadPoolDevice*, const Tensor1&, const Tensor1&);

Tensor1 calculate_spearman_ranks(const Tensor1&);

Correlation logistic_correlation_vector_vector_spearman(const ThreadPoolDevice*, const Tensor1&, const Tensor1&);

Correlation correlation_spearman(const ThreadPoolDevice*, const Tensor2&, const Tensor2&);

// Confidence interval

type r_correlation_to_z_correlation(const type);
type z_correlation_to_r_correlation(const type);

Tensor1 confidence_interval_z_correlation(const type, const Index&);


// Time series correlation

Tensor1 autocorrelations(const ThreadPoolDevice*,
                                 const Tensor1&,
                                 const Index&  = 10);

Tensor1 cross_correlations(const ThreadPoolDevice*,
                                   const Tensor1&,
                                   const Tensor1&,
                                   const Index&);

Tensor2 get_correlation_values(const Tensor<Correlation, 2>&);

// Missing values

pair<Tensor1, Tensor1> filter_missing_values_vector_vector(const Tensor1&, const Tensor1&);
pair<Tensor1, Tensor2> filter_missing_values_vector_matrix(const Tensor1&, const Tensor2&);
pair<Tensor1, Tensor2> filter_missing_values_matrix_vector(const Tensor2&, const Tensor1&);
pair<Tensor2, Tensor2> filter_missing_values_matrix_matrix(const Tensor2&, const Tensor2&);

void register_optimization_algorithms();

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
