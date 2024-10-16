//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <string>
#include <sstream>

#include "neural_network.h"
#include "neural_network_forward_propagation.h"

namespace opennn
{

struct TransformerForwardPropagation;
struct TransformerBackPropagation;

class Transformer : public NeuralNetwork
{
public:

    // Constructors

    explicit Transformer();

    explicit Transformer(const Tensor<Index, 1>&);

    explicit Transformer(const initializer_list<Index>&);

    void set(const Tensor<Index, 1>&);

    void set(const initializer_list<Index>&);

    void set(const Index& input_length, const Index& context_length, const Index& input_dimensions, const Index& context_dimension,
             const Index& embedding_depth, const Index& perceptron_depth, const Index& heads_number, const Index& layers_number);

    void set_dropout_rate(const type&);
    void set_input_vocabulary(const Tensor<string, 1>&);
    void set_context_vocabulary(const Tensor<string, 1>&);

    string calculate_outputs(const string&, const bool& = true);
    Tensor<type, 3> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&);

//    void tokenize_whitespace(const Tensor<string, 1>&, Tensor<type, 2>&);
    void tokenize_wordpiece(const Tensor<string, 1>&, Tensor<type, 2>&);

//    void detokenize_whitespace(Tensor<type, 2>&, ostringstream&);
    void detokenize_wordpiece(Tensor<type, 2>&, ostringstream&);

    void load_transformer(const string&);

protected:

    string name = "transformer";

    Index input_length;

    Index context_length;

    Index input_dimensions;

    Index context_dimension;

    Index embedding_depth;

    Index perceptron_depth;

    Index heads_number;

    Index layers_number;

    type dropout_rate = 0;

    Tensor<string, 1> input_vocabulary;
    Tensor<string, 1> context_vocabulary;

};


struct TransformerForwardPropagation : ForwardPropagation
{
    // Constructors

    TransformerForwardPropagation() {}

    TransformerForwardPropagation(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
    {
        set(new_batch_samples, new_neural_network);
    }

    void set(const Index& new_batch_samples, NeuralNetwork* new_neural_network);

    void print() const;

    Index batch_samples_number = 0;

    Tensor<unique_ptr<LayerForwardPropagation>, 1> layers;
};
};

#endif // TRANSFORMER_H
















