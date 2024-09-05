//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "transformer.h"
#include "tensors.h"
#include "embedding_layer.h"
#include "normalization_layer_3d.h"
#include "multihead_attention_layer.h"
#include "addition_layer_3d.h"
#include "perceptron_layer_3d.h"
#include "probabilistic_layer_3d.h"
#include "strings_utilities.h"

//#include "batch.h"

namespace opennn
{

Transformer::Transformer() : NeuralNetwork()
{
    NeuralNetwork::set();
}


Transformer::Transformer(const Tensor<Index, 1>& architecture)
{
    set(architecture);
}


Transformer::Transformer(const initializer_list<Index>& architecture_list)
{
    set(architecture_list);
}

void Transformer::set(const Tensor<Index, 1>& architecture)
{
    input_length = architecture(0);

    context_length = architecture(1);

    input_dimensions = architecture(2);

    context_dimension = architecture(3);

    embedding_depth = architecture(4);

    perceptron_depth = architecture(5);

    heads_number = architecture(6);

    layers_number = architecture(7);

    set(input_length,
        context_length,
        input_dimensions,
        context_dimension,
        embedding_depth,
        perceptron_depth,
        heads_number,
        layers_number);
}


void Transformer::set(const initializer_list<Index>& architecture_list)
{
    Tensor<Index, 1> architecture(architecture_list.size());
    architecture.setValues(architecture_list);

    set(architecture);
}


void Transformer::set(const Index& input_length, const Index& context_length, const Index& input_dimensions, const Index& context_dimension,
                      const Index& embedding_depth, const Index& perceptron_depth, const Index& heads_number, const Index& layers_number)
{
    delete_layers();
    
    inputs_names.resize(input_length + context_length);


    // Embedding Layers
    
    EmbeddingLayer* input_embedding_layer = new EmbeddingLayer(input_dimensions, input_length, embedding_depth, true);

    input_embedding_layer->set_dropout_rate(dropout_rate);
    input_embedding_layer->set_name("input_embedding");
    add_layer(input_embedding_layer);
    set_layer_inputs_indices("input_embedding", "input");
    

    EmbeddingLayer* context_embedding_layer = new EmbeddingLayer(context_dimension, context_length, embedding_depth, true);

    context_embedding_layer->set_dropout_rate(dropout_rate);
    context_embedding_layer->set_name("context_embedding");
    add_layer(context_embedding_layer);
    set_layer_inputs_indices("context_embedding", "context");

    
    // Encoder

    for(Index i = 0; i < layers_number; i++)
    {
        // Multi head attention

        MultiheadAttentionLayer* context_self_attention_layer =
                new MultiheadAttentionLayer(context_length, context_length, embedding_depth, heads_number);

        context_self_attention_layer->set_dropout_rate(dropout_rate);
        context_self_attention_layer->set_name("context_self_attention_" + to_string(i+1));

        add_layer(context_self_attention_layer);

        if(i == 0)
            set_layer_inputs_indices("context_self_attention_1", {"context_embedding", "context_embedding"});
        else
            set_layer_inputs_indices("context_self_attention_" + to_string(i+1), { "encoder_perceptron_normalization_" + to_string(i), "encoder_perceptron_normalization_" + to_string(i) });

        // Addition

        AdditionLayer3D* context_self_attention_addition_layer = new AdditionLayer3D(context_length, embedding_depth);

        context_self_attention_addition_layer->set_name("context_self_attention_addition_" + to_string(i+1));

        add_layer(context_self_attention_addition_layer);

        if(i == 0)
            set_layer_inputs_indices("context_self_attention_addition_" + to_string(i+1), { "context_embedding", "context_self_attention_" + to_string(i+1) });
        else
            set_layer_inputs_indices("context_self_attention_addition_" + to_string(i+1), { "encoder_perceptron_normalization_" + to_string(i), "context_self_attention_" + to_string(i+1) });

        // Normalization

        NormalizationLayer3D* context_self_attention_normalization_layer = new NormalizationLayer3D(context_length, embedding_depth);

        context_self_attention_normalization_layer->set_name("context_self_attention_normalization_" + to_string(i+1));
        
        add_layer(context_self_attention_normalization_layer);
        
        set_layer_inputs_indices("context_self_attention_normalization_" + to_string(i+1), "context_self_attention_addition_" + to_string(i+1));

        // Perceptron

        PerceptronLayer3D* encoder_internal_perceptron_layer =
                new PerceptronLayer3D(context_length, embedding_depth, perceptron_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        encoder_internal_perceptron_layer->set_name("encoder_internal_perceptron_" + to_string(i+1));
        
        add_layer(encoder_internal_perceptron_layer);
        
        set_layer_inputs_indices("encoder_internal_perceptron_" + to_string(i+1), "context_self_attention_normalization_" + to_string(i+1));

        // Perceptron

        PerceptronLayer3D* encoder_external_perceptron_layer =
            new PerceptronLayer3D(context_length, perceptron_depth, embedding_depth, PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

        encoder_external_perceptron_layer->set_dropout_rate(dropout_rate);
        encoder_external_perceptron_layer->set_name("encoder_external_perceptron_" + to_string(i+1));
        
        add_layer(encoder_external_perceptron_layer);
        
        set_layer_inputs_indices("encoder_external_perceptron_" + to_string(i+1), "encoder_internal_perceptron_" + to_string(i+1));

        // Addition

        AdditionLayer3D* encoder_perceptron_addition_layer = new AdditionLayer3D(context_length, embedding_depth);
        
        encoder_perceptron_addition_layer->set_name("encoder_perceptron_addition_" + to_string(i+1));
        
        add_layer(encoder_perceptron_addition_layer);
        
        set_layer_inputs_indices("encoder_perceptron_addition_" + to_string(i+1), { "context_self_attention_normalization_" + to_string(i+1), "encoder_external_perceptron_" + to_string(i+1) });

        // Normalization

        NormalizationLayer3D* encoder_perceptron_normalization_layer = new NormalizationLayer3D(context_length, embedding_depth);
        
        encoder_perceptron_normalization_layer->set_name("encoder_perceptron_normalization_" + to_string(i+1));
        
        add_layer(encoder_perceptron_normalization_layer);
        
        set_layer_inputs_indices("encoder_perceptron_normalization_" + to_string(i+1), "encoder_perceptron_addition_" + to_string(i+1));
    }
    

    // Decoder

    for(Index i = 0; i < layers_number; i++)
    {
        MultiheadAttentionLayer* input_self_attention_layer =
                new MultiheadAttentionLayer(input_length, input_length, embedding_depth, heads_number, true);

        input_self_attention_layer->set_dropout_rate(dropout_rate);
        input_self_attention_layer->set_name("input_self_attention_" + to_string(i+1));
        add_layer(input_self_attention_layer);

        if(i == 0)
        {
            set_layer_inputs_indices("input_self_attention_1", {"input_embedding", "input_embedding"});
        }
        else
        {
            set_layer_inputs_indices("input_self_attention_" + to_string(i+1), {"decoder_perceptron_normalization_" + to_string(i), "decoder_perceptron_normalization_" + to_string(i)});
        }


        AdditionLayer3D* input_self_attention_addition_layer = new AdditionLayer3D(input_length, embedding_depth);
        input_self_attention_addition_layer->set_name("input_self_attention_addition_" + to_string(i+1));
        add_layer(input_self_attention_addition_layer);
        if(i == 0)
            set_layer_inputs_indices("input_self_attention_addition_" + to_string(i+1), { "input_embedding", "input_self_attention_" + to_string(i+1) });
        else
            set_layer_inputs_indices("input_self_attention_addition_" + to_string(i+1), { "decoder_perceptron_normalization_" + to_string(i), "input_self_attention_" + to_string(i+1) });


        NormalizationLayer3D* input_self_attention_normalization_layer = new NormalizationLayer3D(input_length, embedding_depth);
        input_self_attention_normalization_layer->set_name("input_self_attention_normalization_" + to_string(i+1));
        add_layer(input_self_attention_normalization_layer);
        set_layer_inputs_indices("input_self_attention_normalization_" + to_string(i+1), "input_self_attention_addition_" + to_string(i+1));


        MultiheadAttentionLayer* cross_attention_layer =
                new MultiheadAttentionLayer(input_length, context_length, embedding_depth, heads_number);

        cross_attention_layer->set_dropout_rate(dropout_rate);
        cross_attention_layer->set_name("cross_attention_" + to_string(i+1));
        add_layer(cross_attention_layer);
        set_layer_inputs_indices("cross_attention_" + to_string(i+1), {"input_self_attention_normalization_" + to_string(i+1), "encoder_perceptron_normalization_" + to_string(layers_number)});


        AdditionLayer3D* cross_attention_addition_layer = new AdditionLayer3D(input_length, embedding_depth);
        cross_attention_addition_layer->set_name("cross_attention_addition_" + to_string(i+1));
        add_layer(cross_attention_addition_layer);
        set_layer_inputs_indices("cross_attention_addition_" + to_string(i+1), { "input_self_attention_normalization_" + to_string(i+1), "cross_attention_" + to_string(i+1) });
        

        NormalizationLayer3D* cross_attention_normalization_layer = new NormalizationLayer3D(input_length, embedding_depth);
        cross_attention_normalization_layer->set_name("cross_attention_normalization_" + to_string(i+1));
        add_layer(cross_attention_normalization_layer);
        set_layer_inputs_indices("cross_attention_normalization_" + to_string(i+1), "cross_attention_addition_" + to_string(i+1));


        PerceptronLayer3D* decoder_internal_perceptron_layer =
                new PerceptronLayer3D(input_length, embedding_depth, perceptron_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        decoder_internal_perceptron_layer->set_name("decoder_internal_perceptron_" + to_string(i+1));
        add_layer(decoder_internal_perceptron_layer);
        set_layer_inputs_indices("decoder_internal_perceptron_" + to_string(i+1), "cross_attention_normalization_" + to_string(i+1));


        PerceptronLayer3D* decoder_external_perceptron_layer =
                new PerceptronLayer3D(input_length, perceptron_depth, embedding_depth, PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

        decoder_external_perceptron_layer->set_dropout_rate(dropout_rate);
        decoder_external_perceptron_layer->set_name("decoder_external_perceptron_" + to_string(i+1));
        add_layer(decoder_external_perceptron_layer);
        set_layer_inputs_indices("decoder_external_perceptron_" + to_string(i+1), "decoder_internal_perceptron_" + to_string(i+1));

        AdditionLayer3D* decoder_perceptron_addition_layer = new AdditionLayer3D(input_length, embedding_depth);
        decoder_perceptron_addition_layer->set_name("decoder_perceptron_addition_" + to_string(i+1));
        add_layer(decoder_perceptron_addition_layer);
        set_layer_inputs_indices("decoder_perceptron_addition_" + to_string(i+1), { "cross_attention_normalization_" + to_string(i+1), "decoder_external_perceptron_" + to_string(i+1) });

        NormalizationLayer3D* decoder_perceptron_normalization_layer = new NormalizationLayer3D(input_length, embedding_depth);
        decoder_perceptron_normalization_layer->set_name("decoder_perceptron_normalization_" + to_string(i+1));
        add_layer(decoder_perceptron_normalization_layer);
        set_layer_inputs_indices("decoder_perceptron_normalization_" + to_string(i+1), "decoder_perceptron_addition_" + to_string(i+1));
    }
    
    // Output layer
    
    ProbabilisticLayer3D* final_layer = new ProbabilisticLayer3D(input_length, embedding_depth, input_dimensions);
    
    final_layer->set_name("probabilistic");
    add_layer(final_layer);
    set_layer_inputs_indices("probabilistic", "decoder_perceptron_normalization_" + to_string(layers_number));
    
}


void Transformer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void Transformer::set_input_vocabulary(const Tensor<string, 1>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void Transformer::set_context_vocabulary(const Tensor<string, 1>& new_context_vocabulary)
{
    context_vocabulary = new_context_vocabulary;
}


string Transformer::calculate_outputs(const string& context_string, const bool& imported_vocabulary)
{
    //type start_indicator = 1;
    //type end_indicator = 2;

    //if(imported_vocabulary)
    //{
    type start_indicator = 2;
    type end_indicator = 3;
    //}
    
    const Tensor<Tensor<string, 1>, 1> context_tokens = preprocess_language_documents(tensor_wrapper(context_string));

    const Index batch_samples_number = 1;

    Tensor<type, 2> context(batch_samples_number, context_length);
    context.setZero();
    context(0) = start_indicator;
    
    //if(!imported_vocabulary)    tokenize_whitespace(context_tokens(0), context);
    //else
    tokenize_wordpiece(context_tokens(0), context);
    
    Tensor<type, 2> input(batch_samples_number, input_length);
    input.setZero();
    input(0) = start_indicator;

    ForwardPropagation neural_network_forward_propagation(batch_samples_number, this);
    
    pair<type*, dimensions> context_pair(context.data(), { 1, context_length });
    pair<type*, dimensions> input_pair(input.data(), { 1, input_length });

    Tensor<pair<type*, dimensions>, 1> inputs_pairs(2);

    inputs_pairs(0) = input_pair;
    inputs_pairs(1) = context_pair;

    const Index layers_number = get_layers_number();

    pair<type*, dimensions> outputs_pair = neural_network_forward_propagation.layers(layers_number - 1)->get_outputs_pair();

    TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[1], outputs_pair.second[2]);

    Tensor<type, 1> current_outputs(outputs_pair.second[2]);
    Tensor<Index, 0> prediction;
    
    for(Index i = 1; i < input_length; i++)
    {
        forward_propagate(inputs_pairs, neural_network_forward_propagation);

        current_outputs.device(*thread_pool_device) = outputs.chip(i - 1, 0);

        prediction.device(*thread_pool_device) = current_outputs.argmax();

        input(i) = type(prediction(0));

        if(prediction(0) == end_indicator) break;
    }
    
    ostringstream output_string;

    //if(!imported_vocabulary)    detokenize_whitespace(input, output_string);
    //else
    detokenize_wordpiece(input, output_string);

    return output_string.str();
    
}

Tensor<type, 3> Transformer::calculate_outputs(const Tensor<type, 2>& input, const Tensor<type, 2>& context)
{
    pair<type*, dimensions> input_pair((type*)input.data(), { input.dimension(0), input.dimension(1) });
    pair<type*, dimensions> context_pair((type*)context.data(), { input.dimension(0), context.dimension(1) });

    Tensor<pair<type*, dimensions>, 1> inputs_pair(2);
    inputs_pair(0) = input_pair;
    inputs_pair(1) = context_pair;

    ForwardPropagation forward_propagation(input.dimension(0), this);

    forward_propagate(inputs_pair, forward_propagation, false);

    pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    TensorMap<Tensor<type, 3>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1], outputs_pair.second[2]);

    return outputs;
}


// void Transformer::tokenize_whitespace(const Tensor<string, 1>& context_tokens, Tensor<type, 2>& context)
// {
//     const Index context_vocabulary_size = context_vocabulary.size();

//     bool line_ended = false;

//     for(Index j = 0; j < context_length - 1; j++)
//     {
//         if(j < context_tokens.size())
//         {
//             auto it = find(context_vocabulary.data(), context_vocabulary.data() + context_vocabulary_size, context_tokens(j));

//             const Index word_index = it - context_vocabulary.data();

//             context(j + 1) = type(word_index);
//         }
//         else
//         {
//             if(j == context_tokens.size() || (j == context_length - 2 && !line_ended))
//             {
//                 context(j + 1) = 2; // end indicator
//                 line_ended = true;
//             }
//             else
//             {
//                 break;
//             }
//         }
//     }
// }


void Transformer::tokenize_wordpiece(const Tensor<string, 1>& context_tokens, Tensor<type, 2>& context)
{
    unordered_map<std::string, type> context_vocabulary_map;

    for(Index i = 0; i < context_vocabulary.size(); i++)
        context_vocabulary_map[context_vocabulary(i)] = type(i);

    Index token_counter = 1;
    bool line_ended = false;

    string word;
    string wordpiece;
    string rest;

    auto wordpiece_entry = context_vocabulary_map.find("");
    bool tokenized;

    for(Index j = 0; j < context_length - 1; j++)
    {
        if(j < context_tokens.size() && token_counter < context_length - 1)
        {
            word = context_tokens(j);

            wordpiece_entry = context_vocabulary_map.find(word);

            if(wordpiece_entry != context_vocabulary_map.end())
            {
                context(token_counter) = wordpiece_entry->second;
                token_counter++;
                continue;
            }

            tokenized = false;

            for(Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
            {
                if(token_counter == context_length - 1)
                {
                    tokenized = true;
                    break;
                }

                wordpiece = word.substr(0, wordpiece_length);
                wordpiece_entry = context_vocabulary_map.find(wordpiece);

                if(wordpiece_entry != context_vocabulary_map.end())
                {
                    context(token_counter) = wordpiece_entry->second;
                    token_counter++;

                    rest = word.substr(wordpiece_length);

                    if(rest.empty())
                    {
                        tokenized = true;
                        break;
                    }

                    word = "##" + rest;
                    wordpiece_length = word.length() + 1;
                }
            }

            if(!tokenized)
            {
                context(token_counter) = 1; // unknown indicator
                token_counter++;
            }
        }
        else
        {
            if(j == context_tokens.size()
            || (token_counter == context_length - 1 && !line_ended))
            {
                context(token_counter) = 3; // end indicator
                token_counter++;
                line_ended = true;
            }
            else
            {
                break;
            }
        }
    }
}


// void Transformer::detokenize_whitespace(Tensor<type, 2>& predictions, ostringstream& output_string)
// {
//     for(Index i = 1; i < input_length; i++)
//     {
//         if(predictions(i) == 2)   break;

//         output_string << input_vocabulary(Index(predictions(i))) << " ";
//     }
// }


void Transformer::detokenize_wordpiece(Tensor<type, 2>& predictions, ostringstream& output_string)
{
    output_string << input_vocabulary(Index(predictions(1)));

    string current_prediction;

    for(Index i = 2; i < input_length; i++)
    {
        if(predictions(i) == 3)   break;

        current_prediction = input_vocabulary(Index(predictions(i)));

        if(current_prediction.substr(0, 2) == "##")
        {
            output_string << current_prediction.substr(2);
        }
        else
        {
            output_string << " " << current_prediction;
        }
    }
}


void Transformer::load_transformer(const string& path)
{
    cout << "Loading transformer model..." << endl;

    load(path);

    MultiheadAttentionLayer* cross_attention_layer = static_cast<MultiheadAttentionLayer*>(get_layer("cross_attention_1"));

    const Index new_context_length = cross_attention_layer->get_context_size();
    const Index new_input_length = cross_attention_layer->get_input_size();

    context_length = new_context_length;
    input_length = new_input_length;
}


TransformerForwardPropagation::~TransformerForwardPropagation()
{
    const Index layers_number = layers.size();

    for(Index i = 0; i < layers_number; i++)
    {
        delete layers(i);
    }
}


void TransformerForwardPropagation::set(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
{
    Transformer* neural_network = static_cast<Transformer*>(new_neural_network);

    batch_samples_number = new_batch_samples;

    const Tensor<Layer*, 1> neural_network_layers = neural_network->get_layers();

    const Index layers_number = layers.size();

    layers.resize(layers_number);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers(i)->get_type())
        {

        case Layer::Type::Embedding:
        {
            layers(i) = new EmbeddingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::MultiheadAttention:
        {
            layers(i) = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Perceptron3D:
        {
            layers(i) = new PerceptronLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Probabilistic3D:
        {
            layers(i) = new ProbabilisticLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        default: break;
        }
    }
}


void TransformerForwardPropagation::print() const
{
    cout << "Transformer forward propagation" << endl;

    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << ": " << layers(i)->layer->get_name() << endl;

        layers(i)->print();
    }
}

};



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
