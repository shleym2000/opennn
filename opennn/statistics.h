//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

struct Descriptives
{
    Descriptives(const type& = type(NAN), const type& = type(NAN), const type& = type(NAN), const type& = type(NAN));

    Tensor1 to_tensor() const;

    void set(const type& = type(NAN), const type& = type(NAN), const type& = type(NAN), const type& = type(NAN));

    void save(const filesystem::path&) const;

    void print(const string& = "Descriptives:") const;

    string name = "Descriptives";

    type minimum = type(-1.0);

    type maximum = type(1);

    type mean = type(0);

    type standard_deviation = type(1);

};


struct BoxPlot
{
    BoxPlot(const type& = type(NAN),
            const type& = type(NAN),
            const type& = type(NAN),
            const type& = type(NAN),
            const type& = type(NAN));

    void set(const type& = type(NAN),
             const type& = type(NAN),
             const type& = type(NAN),
             const type& = type(NAN),
             const type& = type(NAN));

    type minimum = type(NAN);

    type first_quartile = type(NAN);

    type median = type(NAN);

    type third_quartile = type(NAN);

    type maximum = type(NAN);
};


struct Histogram
{
    Histogram(const Index& = 0);

    Histogram(const Tensor1&, const Tensor<Index, 1>&);

    Histogram(const Tensor<Index, 1>&, const Tensor1&, const Tensor1&, const Tensor1&);

    Histogram(const Tensor1&, const Index&);

    Histogram(const Tensor1&);

    // Methods

    Index get_bins_number() const;

    Index count_empty_bins() const;

    Index calculate_minimum_frequency() const;

    Index calculate_maximum_frequency() const;

    Index calculate_most_populated_bin() const;

    Tensor1 calculate_minimal_centers() const;

    Tensor1 calculate_maximal_centers() const;

    Index calculate_bin(const type&) const;

    Index calculate_frequency(const type&) const;

    void save(const filesystem::path&) const;

    Tensor1 minimums;

    Tensor1 maximums;

    Tensor1 centers;

    Tensor<Index, 1> frequencies;
};

// Minimum

 type minimum(const Tensor1&);
 type minimum(const Tensor1&, const vector<Index>&);
 Index minimum(const Tensor<Index, 1>&);
 type minimum(const Tensor2&);
 Tensor1 column_minimums(const Tensor2&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());

 // Maximum

 type maximum(const Tensor1&);
 type maximum(const Tensor1&, const vector<Index>&);
 Index maximum(const Tensor<Index, 1>&);
 //type maximum(const Tensor2&);
 Tensor1 column_maximums(const Tensor2&, 
	                             const vector<Index>& = vector<Index>(), 
	                             const vector<Index>& = vector<Index>());

 // Range
 type range(const Tensor1&);

 // Mean
 type mean(const Tensor1&);
 type mean(const Tensor1&, const Index&, const Index&);
 type mean(const Tensor2&,  const Index&);
 Tensor1 mean(const Tensor2&);
 Tensor1 mean(const Tensor2&, const vector<Index>&);
 Tensor1 mean(const Tensor2&, const vector<Index>&, const vector<Index>&);

 // Median
 type median(const Tensor1&);
 type median(const Tensor2&, const Index&);
 Tensor1 median(const Tensor2&);
 Tensor1 median(const Tensor2&, const vector<Index>&);
 Tensor1 median(const Tensor2&, const vector<Index>&, const vector<Index>&);

 // Variance
 type variance(const Tensor1&);
 type variance(const Tensor1&, const Tensor<Index, 1>&);

 // Standard deviation
 type standard_deviation(const Tensor1&);
 //type standard_deviation(const Tensor1&, const Tensor<Index, 1>&);
 Tensor1 standard_deviation(const Tensor1&, const Index&);

 // Assymetry
 type asymmetry(const Tensor1&);

 // Kurtosis
 type kurtosis(const Tensor1&);

 // Quartiles
 Tensor1 quartiles(const Tensor1&);
 Tensor1 quartiles(const Tensor1&, const vector<Index>&);

 // Box plot
 BoxPlot box_plot(const Tensor1&);
 BoxPlot box_plot(const Tensor1&, const vector<Index>&);

 // Descriptives vector
 Descriptives vector_descriptives(const Tensor1&);

 // Descriptives matrix
 vector<Descriptives> descriptives(const Tensor2&);
 vector<Descriptives> descriptives(const Tensor2&, const vector<Index>&, const vector<Index>&);

 // Histograms
 Histogram histogram(const Tensor1&, const Index&  = 10);
 Histogram histogram_centered(const Tensor1&, const type& = type(0), const Index&  = 10);
 Histogram histogram(const Tensor<bool, 1>&);
 Histogram histogram(const Tensor<Index, 1>&, const Index&  = 10);
 vector<Histogram> histograms(const Tensor2&, const Index& = 10);
 Tensor<Index, 1> total_frequencies(const vector<Histogram>&);

 // Minimal indices
 Index minimal_index(const Tensor1&);
 Tensor<Index, 1> minimal_indices(const Tensor1&, const Index&);
 Tensor<Index, 1> minimal_indices(const Tensor2&);

 // Maximal indices
 Index maximal_index(const Tensor1&);
 Tensor<Index, 1> maximal_indices(const Tensor1&, const Index&);
 Tensor<Index, 1> maximal_indices(const Tensor2&);

 // Percentiles
 Tensor1 percentiles(const Tensor1&);
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
