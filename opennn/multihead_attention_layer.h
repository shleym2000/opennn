//  OpenNN: Open Neural Networks Library
//  www.opennn.net
//
//  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S   H E A D E R
//
//  Artificial Intelligence Techniques SL
//  artelnics@artelnics.com

#ifndef MULTIHEADATTENTIONLAYER_H
#define MULTIHEADATTENTIONLAYER_H

#include "layer.h"

namespace opennn
{

class MultiHeadAttention final : public Layer
{

public:

    MultiHeadAttention(const dimensions& = dimensions({0,0}),
                       const Index& = 0,
                       const string& = string());

    MultiHeadAttention(const dimensions&,
                       const dimensions&,
                       const Index& = 0,
                       const string& = string());

    type* link_parameters(type* ptr) override
    {
        auto link_block = [&](type* p, type*& w, type*& b, Index dim) {
            w = p;
            p += (dim * dim);
            p = (type*)(((size_t)p + 63) & ~63);
            b = p;
            p += dim;
            return (type*)(((size_t)p + 63) & ~63);
        };
/*
        Index e = get_embedding_dimension();
        ptr = link_block(ptr, q_weights_ptr, q_biases_ptr, e);
        ptr = link_block(ptr, k_weights_ptr, k_biases_ptr, e);
        ptr = link_block(ptr, v_weights_ptr, v_biases_ptr, e);
        ptr = link_block(ptr, proj_weights_ptr, proj_biases_ptr, e);
*/
        return ptr;
    }

    Index get_query_sequence_length() const;
    Index get_source_sequence_length() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;
    Index get_head_dimension() const;

    type get_scaling_factor() const;

    dimensions get_input_dimensions() const override;

    dimensions get_output_dimensions() const override;

    vector<TensorView> get_parameter_views() const override;

    void set(const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const Index& = 0,
             const bool& = false,
             const string& = "multihead_attention_layer");

    void set_dropout_rate(const type&);

    void apply_causal_mask(Tensor4&) const;

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void print() const override;

    void to_XML(XMLPrinter&) const override;
    void from_XML(const XMLDocument&) override;

    void apply_key_padding_mask(const Tensor<bool, 2>&,Tensor4&) const;

#ifdef OPENNN_CUDA
        // @todo
#endif

private:

    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;

    TensorView query_weights;
    TensorView query_biases;

    TensorView key_weights;
    TensorView key_biases;

    TensorView value_weights;
    TensorView value_biases;

    TensorView projection_weights;
    TensorView projection_biases;

    bool use_causal_mask = false;

    Tensor2 causal_mask;
    Tensor<bool,2> key_mask; // Starting to implement (should be used before softmax so that the probability of the padding is zero)

    type dropout_rate = type(0);

    const type minus_inf = -numeric_limits<float>::infinity();
};


struct MultiHeadAttentionForwardPropagation final : LayerForwardPropagation
{
    MultiHeadAttentionForwardPropagation(const Index& new_batch_size = 0,
                                         Layer* new_layer = nullptr);

    TensorView get_output_view() const override;

    void initialize() override;

    void print() const override;

    Tensor4 query;
    Tensor4 key;
    Tensor4 value;

    Tensor4 attention_weights;
    Tensor4 attention_outputs;

    Tensor3 concatenated_attention_outputs;

    Tensor3 outputs;
};


struct MultiHeadAttentionBackPropagation final : LayerBackPropagation
{
    MultiHeadAttentionBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<TensorView> get_input_derivative_views() const override;

    vector<ParameterView> get_parameter_delta_views() const override;

    void initialize() override;

    void print() const override;

    Tensor4 attention_weight_deltas;
    Tensor4 attention_output_deltas;
    Tensor3 concatenated_attention_output_deltas;

    Tensor4 query_deltas;
    Tensor4 key_deltas;
    Tensor4 value_deltas;

    Tensor2 query_weight_deltas;
    Tensor2 key_weight_deltas;
    Tensor2 value_weight_deltas;

    Tensor2 projection_weight_deltas;

    Tensor1 query_bias_deltas;
    Tensor1 key_bias_deltas;
    Tensor1 value_bias_deltas;
    Tensor1 projection_bias_deltas;

    Tensor1 aux_rows;

    Tensor3 input_query_deltas;
    Tensor3 input_source_deltas;

    Tensor4 softmax_deltas;
};

#ifdef OPENNN_CUDA
    // @todo
#endif

} // namespace opennn


#endif // MULTIHEAD_ATTENTION_LAYER_H


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
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
