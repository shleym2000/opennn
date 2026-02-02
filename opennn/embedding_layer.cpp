//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "embedding_layer.h"

namespace opennn
{

Embedding::Embedding(const dimensions& new_input_dimensions,
                     const Index& new_embedding_dimension,
                     const string& new_label) : Layer()
{
    set(new_input_dimensions[0], new_input_dimensions[1], new_embedding_dimension, new_label);

    name = "Embedding";
}


Index Embedding::get_vocabulary_size() const
{
    return weights.dims[0];
}


Index Embedding::get_sequence_length() const
{
    return sequence_length;
}


Index Embedding::get_embedding_dimension() const
{
    return weights.dims[1];
}


dimensions Embedding::get_input_dimensions() const
{
    return { sequence_length };
}


dimensions Embedding::get_output_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


vector<TensorView*> Embedding::get_parameter_views()
{
    return {&weights};
}


void Embedding::set(const Index new_vocabulary_size,
                    const Index& new_sequence_length,
                    const Index& new_embedding_dimension,
                    const string& new_label)
{
    sequence_length = new_sequence_length;
    label = new_label;

    weights.dims = {new_vocabulary_size, new_embedding_dimension};

    positional_encoding.resize(sequence_length, new_embedding_dimension);
    positional_encoding.setZero();

    const type half_depth = type(new_embedding_dimension)/2;

#pragma omp parallel for collapse(2)

    for(Index i = 0; i < sequence_length; i++)
        for(Index j = 0; j < new_embedding_dimension; j++)
            positional_encoding(i, j) = (j < Index(half_depth))
                ? sin(i / pow(10000, j / half_depth))
                : cos(i / pow(10000, (j - Index(half_depth)) / half_depth));
}

void Embedding::set_scale_embedding(const bool& new_scale_embedding)
{
    scale_embedding = new_scale_embedding;
}


void Embedding::set_add_positional_encoding(const bool& new_add_positional_encoding)
{
    add_positional_encoding = new_add_positional_encoding;
}


void Embedding::set_dropout_rate(const type new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void Embedding::set_parameters_random()
{        
    if(weights.size() == 0) return;

    TensorMap2 weights_map = tensor_map<2>(weights);

    const type scale = 0.05;

    weights_map.setRandom();
    weights_map.device(*device) = weights_map * scale;

    weights_map.chip(0, 0).setZero();
}


void Embedding::set_parameters_glorot()
{
    if(weights.size() == 0) return;

    const Index vocabulary_size = weights.dims[0];
    const Index embedding_dimension = weights.dims[1];

    const type limit = sqrt(type(6.0) / (vocabulary_size + embedding_dimension));

    TensorMap2 weights_map = tensor_map<2>(weights);

    weights_map.setRandom();
    weights_map.device(*device) = weights_map * limit;

    weights_map.chip(0, 0).setZero();
}


void Embedding::forward_propagate(const vector<TensorView>& input_views,
                                  unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                  const bool& is_training)
{
    const TensorMap2 inputs = tensor_map<2>(input_views[0]);

    TensorMap3 outputs = tensor_map<3>(layer_forward_propagation->outputs);

    const Index batch_size = outputs.dimension(0);
    const Index embedding_dimension = outputs.dimension(2);

    const type coefficient = sqrt(type(get_embedding_dimension()));

    if (outputs.dimension(0) != batch_size)
        throw runtime_error("Batch size mismatch between inputs and outputs: inputs.dimension(0) = "
                            + to_string(batch_size) + ", outputs.dimension(0) = " + to_string(outputs.dimension(0)));

    const TensorMap2 weights_map = tensor_map<2>(weights);

    #pragma omp parallel for
    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        auto sample_output = outputs.chip(sample_index, 0);

        for(Index word_index = 0; word_index < sequence_length; word_index++)
        {
            const Index token_id = inputs(sample_index, word_index);

            if (token_id < 0 || token_id >= weights.dims[0])
                throw runtime_error("Invalid token_id \n");

            const auto embedding = weights_map.chip(token_id, 0);

            scale_embedding
                ? sample_output.chip(word_index, 0) = embedding*coefficient
                : sample_output.chip(word_index, 0) = embedding;
        }
    }

    if(add_positional_encoding)
        outputs.device(*device) += positional_encoding
                                  .reshape(array_3(1, sequence_length, embedding_dimension))
                                  .broadcast(array_3(batch_size, 1, 1));


    //if(is_training && dropout_rate > 0)
    //    dropout(outputs, dropout_rate);
}


void Embedding::back_propagate(const vector<TensorView>& input_views,
                               const vector<TensorView>& delta_views,
                               unique_ptr<LayerForwardPropagation>&,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index embedding_dimension = get_embedding_dimension();
    const Index batch_size = input_views[0].dims[0];
    const Index sequence_length = input_views[0].dims[1];

    const TensorMap2 inputs = tensor_map<2>(input_views[0]);

    if (delta_views.size() > 1)
        add_deltas(delta_views);

    TensorMap3 deltas = tensor_map<3>(delta_views[0]);

    // Back propagation

    EmbeddingBackPropagation* embedding_back_propagation =
        static_cast<EmbeddingBackPropagation*>(back_propagation.get());

    TensorMap2 weight_deltas = tensor_map<2>(embedding_back_propagation->weight_deltas);
    weight_deltas.setZero();

    if(scale_embedding)
        deltas.device(*device) = deltas*sqrt(type(embedding_dimension));

    #pragma omp parallel for
    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        auto sample_deltas = deltas.chip(sample_index, 0);

        for(Index word_index = 0; word_index < sequence_length; word_index++)
            weight_deltas.chip(Index(inputs(sample_index, word_index)), 0)
                += sample_deltas.chip(word_index, 0);
    }

    weight_deltas.chip(0, 0).setZero(); // PAD
}


void Embedding::print() const
{
    cout << "Embedding Layer" << endl
         << "Label: " << label << endl
         << "Type: Embedding" << endl
         << "Input dimensions: " << get_input_dimensions() << endl
         << "Output dimensions: " << get_output_dimensions() << endl
         << "Vocabulary size: " << get_vocabulary_size() << endl
         << "Sequence length: " << get_sequence_length() << endl
         << "Embedding dimension: " << get_embedding_dimension() << endl
         << "Dropout rate: " << dropout_rate << endl
         << "Weights dimensions: " << weights.dims << endl;

    cout << "Weights:\n " << weights.dims << endl;
    cout << "Positional encoding:\n" << positional_encoding << endl;
}


void Embedding::from_XML(const XMLDocument& document)
{
    const XMLElement* embedding_layer_element = document.FirstChildElement("Embedding");

    if(!embedding_layer_element)
        throw runtime_error("Embedding element is nullptr.\n");

    const string new_name = read_xml_string(embedding_layer_element, "Name");
    const Index new_vocabulary_size = read_xml_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_xml_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_name);
}


void Embedding::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Embedding");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "VocabularySize", to_string(get_vocabulary_size()));
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingSize", to_string(get_embedding_dimension()));

    printer.CloseElement();
}


EmbeddingForwardPropagation::EmbeddingForwardPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void EmbeddingForwardPropagation::initialize()
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);

    const Index sequence_length = embedding_layer->get_sequence_length();
    const Index embedding_dimension = embedding_layer->get_embedding_dimension();

    outputs.dims = {batch_size, sequence_length, embedding_dimension};
}


void EmbeddingForwardPropagation::print() const
{
    cout << "Output dimensions:" << endl;
    //       cout << output_dimensions << endl;
    cout << "Outputs:" << endl;
    //       cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
}


EmbeddingBackPropagation::EmbeddingBackPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void EmbeddingBackPropagation::initialize()
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);

    const Index embedding_dimension = embedding_layer->get_embedding_dimension();
    const Index vocabulary_size = embedding_layer->get_vocabulary_size();

    weight_deltas.dims = {vocabulary_size, embedding_dimension};
}


vector<TensorView*> EmbeddingBackPropagation::get_workspace_views()
{
    return {&weight_deltas};
}


void EmbeddingBackPropagation::print() const
{
}

#ifdef OPENNN_CUDA

void Embedding::forward_propagate_cuda(const vector<TensorViewCuda>& inputs_device,
                                       unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                       const bool& is_training)
{
    throw runtime_error("Embedding::forward_propagate_cuda is not yet implemented. Please check back in a future version.");
}


void Embedding::back_propagate_cuda(const vector<TensorViewCuda>&,
                                    const vector<TensorViewCuda>&,
                                    unique_ptr<LayerForwardPropagationCuda>&,
                                    unique_ptr<LayerBackPropagationCuda>&) const
{
    throw runtime_error("Embedding::back_propagate_cuda is not yet implemented. Please check back in a future version.");
}


vector<TensorViewCuda*> Embedding::get_parameter_views_device()
{
    throw runtime_error("Embedding::get_parameter_views_device is not yet implemented. Please check back in a future version.");
    return vector<TensorViewCuda*>();
}


EmbeddingForwardPropagationCuda::EmbeddingForwardPropagationCuda(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void EmbeddingForwardPropagationCuda::initialize()
{
    throw runtime_error("EmbeddingForwardPropagationCuda::initialize is not yet implemented. Please check back in a future version.");
}


void EmbeddingForwardPropagationCuda::print() const
{
    throw runtime_error("EmbeddingForwardPropagationCuda::print is not yet implemented. Please check back in a future version.");
}


EmbeddingBackPropagationCuda::EmbeddingBackPropagationCuda(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void EmbeddingBackPropagationCuda::initialize()
{
    throw runtime_error("EmbeddingBackPropagationCuda::initialize is not yet implemented. Please check back in a future version.");
}


vector<TensorViewCuda*> EmbeddingBackPropagationCuda::get_workspace_views_device()
{
    return {&weight_deltas_device};
}


void EmbeddingBackPropagationCuda::print() const
{
    throw runtime_error("EmbeddingBackPropagationCuda::print is not yet implemented. Please check back in a future version.");
}

REGISTER(LayerForwardPropagationCuda, EmbeddingForwardPropagationCuda, "Embedding")
REGISTER(LayerBackPropagationCuda, EmbeddingBackPropagationCuda, "Embedding")

#endif

REGISTER(Layer, Embedding, "Embedding")
REGISTER(LayerForwardPropagation, EmbeddingForwardPropagation, "Embedding")
REGISTER(LayerBackPropagation, EmbeddingBackPropagation, "Embedding")

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
