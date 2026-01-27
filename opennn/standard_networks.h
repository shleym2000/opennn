//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S  C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "neural_network.h"

namespace opennn
{

class ApproximationNetwork : public NeuralNetwork
{

public:

    ApproximationNetwork(const dimensions& input_dimensions,
                         const dimensions& complexity_dimensions,
                         const dimensions& output_dimensions);
};


class ClassificationNetwork : public NeuralNetwork
{

public:

    ClassificationNetwork(const dimensions& input_dimensions,
                          const dimensions& complexity_dimensions,
                          const dimensions& output_dimensions);
};


class ForecastingNetwork : public NeuralNetwork
{

public:

    ForecastingNetwork(const dimensions& input_dimensions,
                       const dimensions& complexity_dimensions,
                       const dimensions& output_dimensions);
};


class AutoAssociationNetwork : public NeuralNetwork
{

public:

    AutoAssociationNetwork(const dimensions& input_dimensions,
                           const dimensions& complexity_dimensions,
                           const dimensions& output_dimensions);
};


class ImageClassificationNetwork : public NeuralNetwork
{

public:

    ImageClassificationNetwork(const dimensions& input_dimensions,
                               const dimensions& complexity_dimensions,
                               const dimensions& output_dimensions);
};


class SimpleResNet : public NeuralNetwork
{

public:

    SimpleResNet(const dimensions& input_dimensions,
                 const vector<Index>& blocks_per_stage,
                 const dimensions& initial_filters,
                 const dimensions& output_dimensions);
};


class VGG16 final : public NeuralNetwork
{
public:

    VGG16(const dimensions& input_dimensions, const dimensions& target_dimensions);

    VGG16(const filesystem::path&);

    void set(const dimensions& input_dimensions, const dimensions& target_dimensions);

};


class TextClassificationNetwork : public NeuralNetwork
{

public:

    TextClassificationNetwork(const dimensions& input_dimensions,
                              const dimensions& complexity_dimensions,
                              const dimensions& output_dimensions,
                              const vector<string>& new_input_vocabulary = vector<string>());
};


class Transformer final : public NeuralNetwork
{
public:

    Transformer(const Index& = 0,
                const Index& = 0,
                const Index& = 0,
                const Index& = 0,
                const Index& = 0,
                const Index& = 0,
                const Index& = 0,
                const Index& = 0);

    void set(const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const Index& = 0);

    Index get_input_sequence_length() const;
    Index get_decoder_sequence_length() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;

    void set_dropout_rate(const type&);
    void set_input_vocabulary(const vector<string>&);
    void set_output_vocabulary(const vector<string>&);

    string calculate_outputs(const string&);

private:

    unordered_map<string, Index> input_vocabulary_map;
    unordered_map<Index, string> output_inverse_vocabulary_map;
};

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
