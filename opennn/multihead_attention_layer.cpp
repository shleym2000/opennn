//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "strings_utilities.h"
#include "tensors.h"
#include "multihead_attention_layer.h"

namespace opennn
{

MultiheadAttentionLayer::MultiheadAttentionLayer() : Layer()
{
    set();

    layer_type = Type::MultiheadAttention;
}


MultiheadAttentionLayer::MultiheadAttentionLayer(const Index& new_input_size,
                                                 const Index& new_context_size,
                                                 const Index& new_depth,
                                                 const Index& new_heads_number,
                                                 const bool& use_causal_mask) : Layer()
{
    set(new_input_size, new_context_size, new_depth, new_heads_number);

    set_causal_mask(use_causal_mask);

    layer_type = Type::MultiheadAttention;

    name = "multihead_attention_layer";
}


Index MultiheadAttentionLayer::get_input_size() const
{
    return input_size;
}


Index MultiheadAttentionLayer::get_context_size() const
{
    return context_size;
}


Index MultiheadAttentionLayer::get_depth() const
{
    return depth;
}


Index MultiheadAttentionLayer::get_heads_number() const
{
    return heads_number;
}


Index MultiheadAttentionLayer::get_weights_depth() const
{
    return hidden_depth;
}


dimensions MultiheadAttentionLayer::get_output_dimensions() const
{
    return { input_size, depth };
}


Tensor<type, 3> MultiheadAttentionLayer::get_query_weights() const
{
    return query_weights;
}


Tensor<type, 2> MultiheadAttentionLayer::get_query_biases() const
{
    return query_biases;
}


Tensor<type, 3> MultiheadAttentionLayer::get_key_weights() const
{
    return key_weights;
}


Tensor<type, 2> MultiheadAttentionLayer::get_key_biases() const
{
    return key_biases;
}


Tensor<type, 3> MultiheadAttentionLayer::get_value_weights() const
{
    return value_weights;
}


Tensor<type, 2> MultiheadAttentionLayer::get_value_biases() const
{
    return value_biases;
}


Tensor<type, 3> MultiheadAttentionLayer::get_projection_weights() const
{
    return projection_weights;
}


Tensor<type, 1> MultiheadAttentionLayer::get_projection_biases() const
{
    return projection_biases;
}


Index MultiheadAttentionLayer::get_parameters_number() const
{
    return
        query_weights.size() + query_biases.size()
        + key_weights.size() + key_biases.size()
        + value_weights.size() + value_biases.size()
        + projection_weights.size() + projection_biases.size();
}


Tensor<type, 1> MultiheadAttentionLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    Index parameters_index = 0;

    memcpy(parameters.data(), query_weights.data(), query_weights.size()*sizeof(type));

    parameters_index += query_weights.size();
   
    memcpy(parameters.data() + parameters_index, query_biases.data(), query_biases.size()*sizeof(type));

    parameters_index += query_biases.size();
   
    memcpy(parameters.data() + parameters_index, key_weights.data(), key_weights.size()*sizeof(type));

    parameters_index += key_weights.size();

    memcpy(parameters.data() + parameters_index, key_biases.data(), key_biases.size()*sizeof(type));

    parameters_index += key_biases.size();

    memcpy(parameters.data() + parameters_index, value_weights.data(), value_weights.size()*sizeof(type));

    parameters_index += value_weights.size();

    memcpy(parameters.data() + parameters_index, value_biases.data(), value_biases.size()*sizeof(type));

    parameters_index += value_biases.size();

    memcpy(parameters.data() + parameters_index, projection_weights.data(), projection_weights.size()*sizeof(type));

    parameters_index += projection_weights.size();

    memcpy(parameters.data() + parameters_index, projection_biases.data(), projection_biases.size()*sizeof(type));

    return parameters;
}


const bool& MultiheadAttentionLayer::get_display() const
{
    return display;
}


void MultiheadAttentionLayer::set()
{
    input_size = 0;

    depth = 0;

    heads_number = 0;

    query_weights.resize(0, 0, 0);
    query_biases.resize(0, 0);

    key_weights.resize(0, 0, 0);
    key_biases.resize(0, 0);

    value_weights.resize(0, 0, 0);
    value_biases.resize(0, 0);

    projection_weights.resize(0, 0, 0);
    projection_biases.resize(0);

    set_default();
}


void MultiheadAttentionLayer::set(const Index& new_input_size,
                                  const Index& new_context_size,
                                  const Index& new_depth,
                                  const Index& new_heads_number)
{
    input_size = new_input_size;

    context_size = new_context_size;

    depth = new_depth;

    heads_number = new_heads_number;

    set_weights();

    scaling_factor = type(1) / type(sqrt(hidden_depth));

    set_default();
}


void MultiheadAttentionLayer::set_default()
{
    name = "multihead_attention_layer";

    display = true;

    layer_type = Type::MultiheadAttention;

    dropout_rate = 0;
}



void MultiheadAttentionLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const type* new_parameters_data = new_parameters.data();

    type* query_weights_data = query_weights.data();
    type* query_biases_data = query_biases.data();

    type* key_weights_data = key_weights.data();
    type* key_biases_data = key_biases.data();

    type* value_weights_data = value_weights.data();
    type* value_biases_data = value_biases.data();

    type* projection_weights_data = projection_weights.data();
    type* projection_biases_data = projection_biases.data();

    Index parameters_index = index;

    memcpy(query_weights_data, new_parameters_data + parameters_index, query_weights.size()*sizeof(type));

    parameters_index += query_weights.size();

    memcpy(query_biases_data, new_parameters_data + parameters_index, query_biases.size()*sizeof(type));

    parameters_index += query_biases.size();

    memcpy(key_weights_data, new_parameters_data + parameters_index, key_weights.size()*sizeof(type));

    parameters_index += key_weights.size();

    memcpy(key_biases_data, new_parameters_data + parameters_index, key_biases.size()*sizeof(type));

    parameters_index += key_biases.size();

    memcpy(value_weights_data, new_parameters_data + parameters_index, value_weights.size()*sizeof(type));

    parameters_index += value_weights.size();

    memcpy(value_biases_data, new_parameters_data + parameters_index, value_biases.size()*sizeof(type));

    parameters_index += value_biases.size();

    memcpy(projection_weights_data, new_parameters_data + parameters_index, projection_weights.size()*sizeof(type));

    parameters_index += projection_weights.size();

    memcpy(projection_biases_data, new_parameters_data + parameters_index, projection_biases.size()*sizeof(type));
}


void MultiheadAttentionLayer::set_input_size(const Index& new_input_size)
{
    input_size = new_input_size;
}


void MultiheadAttentionLayer::set_context_size(const Index& new_context_size)
{
    context_size = new_context_size;
}


void MultiheadAttentionLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;
}


void MultiheadAttentionLayer::set_heads_number(const Index& new_heads_number)
{
    heads_number = new_heads_number;

    set_weights();
}


void MultiheadAttentionLayer::set_weights()
{
    hidden_depth = Index(depth / heads_number); //depth;

    query_weights.resize(depth, hidden_depth, heads_number);
    query_biases.resize(hidden_depth, heads_number);

    key_weights.resize(depth, hidden_depth, heads_number);
    key_biases.resize(hidden_depth, heads_number);

    value_weights.resize(depth, hidden_depth, heads_number);
    value_biases.resize(hidden_depth, heads_number);

    projection_weights.resize(hidden_depth, depth, heads_number);
    projection_biases.resize(depth);

    set_parameters_glorot();
}


void MultiheadAttentionLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

    set_random(query_weights, minimum, maximum);

    set_random(query_biases, minimum, maximum);

    set_random(key_weights, minimum, maximum);

    set_random(key_biases, minimum, maximum);

    set_random(value_weights, minimum, maximum);

    set_random(value_biases, minimum, maximum);

    set_random(projection_weights, minimum, maximum);

    set_random(projection_biases, minimum, maximum);
}


void MultiheadAttentionLayer::set_parameters_glorot()
{
    query_biases.setZero();
    key_biases.setZero();
    value_biases.setZero();
    projection_biases.setZero();

    const type limit = sqrt(6 / type(depth + hidden_depth * heads_number));

    const type minimum = -limit;
    const type maximum = limit;

    set_random(query_weights, minimum, maximum);

    set_random(key_weights, minimum, maximum);

    set_random(value_weights, minimum, maximum);

    set_random(projection_weights, minimum, maximum);
}


void MultiheadAttentionLayer::set_parameters_constant(const type& value)
{
    query_weights.setConstant(value);
    query_biases.setConstant(value);

    key_weights.setConstant(value);
    key_biases.setConstant(value);

    value_weights.setConstant(value);
    value_biases.setConstant(value);

    projection_weights.setConstant(value);
    projection_biases.setConstant(value);
}


void MultiheadAttentionLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void MultiheadAttentionLayer::set_causal_mask(const bool& new_use_causal_mask)
{
    if(use_causal_mask && input_size != context_size)

    throw runtime_error("Causal mask can only be applied to self-attention. "
                        "In this case, input size (" + to_string(input_size) + ") "
                        "should be equal to context size (" + to_string(context_size) + ").");

    use_causal_mask = new_use_causal_mask;

    build_causal_mask();
}


void MultiheadAttentionLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void MultiheadAttentionLayer::build_causal_mask()
{
    constexpr type m_inf = -numeric_limits<type>::infinity();

    causal_mask.resize(context_size, input_size);
    causal_mask.setZero();

    #pragma omp parallel for
    for(Index input_index = 0; input_index < input_size; input_index++)
        for(Index context_index = input_index + 1; context_index < context_size; context_index++)
            causal_mask(context_index, input_index) = m_inf;
}


void MultiheadAttentionLayer::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    const Index batch_samples_number = attention_scores.dimension(2);

    #pragma omp parallel for
    
    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        for(Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            type* sample_attention_scores_data = attention_scores.data()
                + sample_index * context_size * input_size
                + head_index * context_size * input_size * batch_samples_number;

            TensorMap<Tensor<type, 2>> sample_attention_scores(sample_attention_scores_data, 
                                                               context_size, 
                                                               input_size);

            sample_attention_scores.device(*thread_pool_device) += causal_mask;
        }
    }
}


void MultiheadAttentionLayer::calculate_transformation(const Tensor<type, 3>& input,
                                                       Tensor<type, 4>& transformed_input,
                                                       const Tensor<type, 3>& weights,
                                                       const Tensor<type, 2>& biases,
                                                       Tensor<type, 2>& sample_matrix) const
{
    // @todo Check if we can do this with transposed matrices in contract. 

    const Index batch_size = input.dimension(0);
    const Index variables_number = input.dimension(1);

    type* weights_data = (type*)weights.data();
    type* biases_data = (type*)biases.data();
    type* transformed_input_data = transformed_input.data();
    
    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        type* head_weights_data = weights_data + head_index * depth * hidden_depth;
        type* head_biases_data = biases_data + head_index * hidden_depth;
        type* head_transformed_input_data = transformed_input_data + head_index * batch_size * variables_number * hidden_depth;

        const TensorMap<Tensor<type, 2>> head_weights(head_weights_data, depth, hidden_depth);
        const TensorMap<Tensor<type, 1>> head_biases(head_biases_data, hidden_depth);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            sample_matrix = input.chip(sample_index, 0);

            type* sample_transformed_input_data = head_transformed_input_data + sample_index * variables_number * hidden_depth;

            TensorMap<Tensor<type, 2>> sample_transformed_input(sample_transformed_input_data, variables_number, hidden_depth);

            sample_transformed_input.device(*thread_pool_device)
                = sample_matrix.contract(head_weights, A_B);

            sum_columns(thread_pool_device, head_biases, sample_transformed_input);
        }
    }
}


void MultiheadAttentionLayer::calculate_output_projection(const Tensor<type, 4>& attention_outputs,
                                                          Tensor<type, 4>& projection_outputs,
                                                          Tensor<type, 3>& outputs) const 
{
    const Index batch_size = outputs.dimension(0);

    type* attention_outputs_data = (type*)attention_outputs.data();
    type* projection_outputs_data = projection_outputs.data();
    type* projection_weights_data = (type*)projection_weights.data();

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        type* head_projection_output_data = projection_outputs_data + head_index * batch_size * input_size * depth;
        type* head_projection_weights_data = projection_weights_data + head_index * hidden_depth * depth;
        type* head_attention_output_data = attention_outputs_data + head_index * input_size * hidden_depth * batch_size;

        TensorMap<Tensor<type, 3>> head_projection_output(head_projection_output_data, batch_size, input_size, depth);

        const TensorMap<Tensor<type, 2>> head_projection_weights(head_projection_weights_data, hidden_depth, depth);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            type* sample_attention_output_data = head_attention_output_data + sample_index * input_size * hidden_depth;

            const TensorMap<Tensor<type, 2>> sample_attention_output(sample_attention_output_data, input_size, hidden_depth);

            head_projection_output.chip(sample_index, 0).device(*thread_pool_device)
                = sample_attention_output.contract(head_projection_weights, A_B);
        }
    }

    outputs.device(*thread_pool_device) = projection_outputs.sum(projection_sum_index);

    sum_matrices(thread_pool_device, projection_biases, outputs);
}


void MultiheadAttentionLayer::compute_attention_scores(const Tensor<type, 4>& query,
                                                       const Tensor<type, 4>& key,
                                                       Tensor<type, 4>& attention_scores,
                                                       Tensor<type, 4>& attention_weights) const
{
    batch_matrix_multiplication(thread_pool_device, key, query, attention_scores, A_BT);

    attention_scores.device(*thread_pool_device) = attention_scores * scaling_factor;

    if(use_causal_mask) apply_causal_mask(attention_scores);
/*
    softmax(attention_scores, attention_weights);
*/
}


void MultiheadAttentionLayer::compute_attention_outputs(const Tensor<type, 4>& value,
                                                       const Tensor<type, 4>& attention_weights,
                                                       Tensor<type, 4>& attention_outputs) const 
{
    batch_matrix_multiplication(thread_pool_device, attention_weights, value, attention_outputs, AT_B);
}


void MultiheadAttentionLayer::dropout(Tensor<type, 4>& attention_scores) const
{
    /*
    const Index batch_samples_number = attention_scores.dimension(2);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type* entry_data;
    type random = 0;
    
#pragma omp parallel for
    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        for(Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            for(Index position_index = 0; position_index < input_size; position_index++)
            {
                entry_data = attention_scores.data()
                             + position_index * context_size
                             + sample_index * input_size * context_size
                             + head_index * batch_samples_number * input_size * context_size;

                TensorMap<Tensor<type, 1>> entry(entry_data, context_size);

                random = calculate_random_uniform((type)0, (type)1);

                random < dropout_rate ? entry.setZero()
                    : entry = entry * scaling_factor;
            }
        }
    }
    */
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    #pragma omp parallel for
    for(Index i = 0; i < attention_scores.size(); i++)
        attention_scores(i) = (calculate_random_uniform(type(0), type(1)) < dropout_rate)
            ? 0
            : attention_scores(i) * scaling_factor;
}


void MultiheadAttentionLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                                const bool& is_training)
{
    MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation =
        static_cast<MultiheadAttentionLayerForwardPropagation*>(layer_forward_propagation.get());

    const TensorMap<Tensor<type, 3>> input = tensor_map_3(input_pairs[0]);

    const TensorMap<Tensor<type, 3>> context = tensor_map_3(input_pairs[1]);

    Tensor<type, 4>& query = multihead_attention_layer_forward_propagation->query;
    Tensor<type, 4>& key = multihead_attention_layer_forward_propagation->key;
    Tensor<type, 4>& value = multihead_attention_layer_forward_propagation->value;

    Tensor<type, 2>& sample_matrix = multihead_attention_layer_forward_propagation->sample_matrix;

    Tensor<type, 4>& attention_scores = multihead_attention_layer_forward_propagation->attention_scores;
    Tensor<type, 4>& attention_weights = multihead_attention_layer_forward_propagation->attention_weights;

    Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    Tensor<type, 4>& projection_outputs = multihead_attention_layer_forward_propagation->projection_outputs;
    Tensor<type, 3>& outputs = multihead_attention_layer_forward_propagation->outputs;

    calculate_transformation(input, query, query_weights, query_biases, sample_matrix);

    calculate_transformation(context, key, key_weights, key_biases, sample_matrix);
    
    calculate_transformation(context, value, value_weights, value_biases, sample_matrix);
    
    compute_attention_scores(query,
                             key,
                             attention_scores,
                             attention_weights);

    if(is_training && dropout_rate > type(0)) 
        dropout(attention_weights);
    
    compute_attention_outputs(value,
                              attention_weights,
                              attention_outputs);

    calculate_output_projection(attention_outputs,
                                projection_outputs,
                                outputs);
}


void MultiheadAttentionLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                             const vector<pair<type*, dimensions>>& delta_pairs,
                                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> input = tensor_map_3(input_pairs[0]);

    const TensorMap<Tensor<type, 3>> context = tensor_map_3(input_pairs[1]);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    const Index batch_samples_number = input_pairs[0].second[0];

    type* query_weights_data = (type*)query_weights.data();
    type* key_weights_data = (type*)key_weights.data();
    type* value_weights_data = (type*)value_weights.data();
    type* projection_weights_data = (type*)projection_weights.data();

    // Forward propagation

    const MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation =
        static_cast<MultiheadAttentionLayerForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 4>& attention_weights = multihead_attention_layer_forward_propagation->attention_weights;
    const Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    const Tensor<type, 4>& query = multihead_attention_layer_forward_propagation-> query;
    const Tensor<type, 4>& key = multihead_attention_layer_forward_propagation->key;
    const Tensor<type, 4>& value = multihead_attention_layer_forward_propagation->value;
    
    type* attention_weights_data = (type*)attention_weights.data();
    type* attention_outputs_data = (type*)attention_outputs.data();

    type* query_data = (type*)query.data();
    type* key_data = (type*)key.data();
    type* value_data = (type*)value.data();

    // Back propagation

    MultiheadAttentionLayerBackPropagation* multihead_attention_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& projection_weights_derivatives = multihead_attention_layer_back_propagation->projection_weights_derivatives;

    Tensor<type, 4>& error_attention_scores_derivatives = multihead_attention_layer_back_propagation->error_attention_scores_derivatives;
    Tensor<type, 4>& error_attention_weights_derivatives = multihead_attention_layer_back_propagation->error_attention_weights_derivatives;
    Tensor<type, 4>& error_attention_output_derivatives = multihead_attention_layer_back_propagation->error_attention_output_derivatives;

    Tensor<type, 2>& sample_deltas = multihead_attention_layer_back_propagation->sample_deltas;

    Tensor<type, 4>& error_query_derivatives = multihead_attention_layer_back_propagation->error_query_derivatives;
    Tensor<type, 4>& error_key_derivatives = multihead_attention_layer_back_propagation->error_key_derivatives;
    Tensor<type, 4>& error_value_derivatives = multihead_attention_layer_back_propagation->error_value_derivatives;

    Tensor<type, 3>& query_weights_derivatives = multihead_attention_layer_back_propagation->query_weights_derivatives;
    Tensor<type, 3>& key_weights_derivatives = multihead_attention_layer_back_propagation->key_weights_derivatives;
    Tensor<type, 3>& value_weights_derivatives = multihead_attention_layer_back_propagation->value_weights_derivatives;

    Tensor<type, 3>& input_derivatives = multihead_attention_layer_back_propagation->input_derivatives;
    input_derivatives.setZero();
    Tensor<type, 3>& context_derivatives = multihead_attention_layer_back_propagation->context_derivatives;
    context_derivatives.setZero();

    Tensor<type, 1>& aux_rows = multihead_attention_layer_back_propagation->aux_rows;

    Tensor<type, 2>& query_biases_derivatives = multihead_attention_layer_back_propagation->query_biases_derivatives;
    Tensor<type, 2>& key_biases_derivatives = multihead_attention_layer_back_propagation->key_biases_derivatives;
    Tensor<type, 2>& value_biases_derivatives = multihead_attention_layer_back_propagation->value_biases_derivatives;
    Tensor<type, 1>& projection_biases_derivatives = multihead_attention_layer_back_propagation->projection_biases_derivatives;

    type* projection_weights_derivatives_data = projection_weights_derivatives.data();

    type* error_attention_scores_derivatives_data = error_attention_scores_derivatives.data();
    type* error_attention_weights_derivatives_data = error_attention_weights_derivatives.data();
    type* error_attention_output_derivatives_data = error_attention_output_derivatives.data();

    type* error_query_derivatives_data = error_query_derivatives.data();
    type* error_key_derivatives_data = error_key_derivatives.data();
    type* error_value_derivatives_data = error_value_derivatives.data();

    type* query_weights_derivatives_data = query_weights_derivatives.data();
    type* key_weights_derivatives_data = key_weights_derivatives.data();
    type* value_weights_derivatives_data = value_weights_derivatives.data();

    type* query_biases_derivatives_data = query_biases_derivatives.data();
    type* key_biases_derivatives_data = key_biases_derivatives.data();
    type* value_biases_derivatives_data = value_biases_derivatives.data();
    
    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        type* head_query_weights_data = query_weights_data + head_index * depth * hidden_depth;
        type* head_key_weights_data = key_weights_data + head_index * depth * hidden_depth;
        type* head_value_weights_data = value_weights_data + head_index * depth * hidden_depth;

        type* head_query_data = query_data + head_index * input_size * hidden_depth * batch_samples_number;
        type* head_key_data = key_data + head_index * context_size * hidden_depth * batch_samples_number;
        type* head_value_data = value_data + head_index * context_size * hidden_depth * batch_samples_number;

        type* head_projection_weights_data = projection_weights_data + head_index * hidden_depth * depth;
        type* head_attention_weights_data = attention_weights_data + head_index * context_size * input_size * batch_samples_number;
        type* head_attention_outputs_data = attention_outputs_data + head_index * input_size * hidden_depth * batch_samples_number;

        type* head_projection_weights_derivatives_data = projection_weights_derivatives_data + head_index * hidden_depth * depth;

        type* head_attention_scores_derivatives_data = error_attention_scores_derivatives_data + head_index * context_size * input_size * batch_samples_number;
        type* head_attention_weights_derivatives_data = error_attention_weights_derivatives_data + head_index * context_size * input_size * batch_samples_number;
        type* head_attention_output_derivatives_data = error_attention_output_derivatives_data + head_index * input_size * hidden_depth * batch_samples_number;

        type* head_query_derivatives_data = error_query_derivatives_data + head_index * input_size * hidden_depth * batch_samples_number;
        type* head_key_derivatives_data = error_key_derivatives_data + head_index * context_size * hidden_depth * batch_samples_number;
        type* head_value_derivatives_data = error_value_derivatives_data + head_index * context_size * hidden_depth * batch_samples_number;

        type* head_query_weights_derivatives_data = query_weights_derivatives_data + head_index * depth * hidden_depth;
        type* head_key_weights_derivatives_data = key_weights_derivatives_data + head_index * depth * hidden_depth;
        type* head_value_weights_derivatives_data = value_weights_derivatives_data + head_index * depth * hidden_depth;
        
        type* head_query_biases_derivatives_data = query_biases_derivatives_data + head_index * hidden_depth;
        type* head_key_biases_derivatives_data = key_biases_derivatives_data + head_index * hidden_depth;
        type* head_value_biases_derivatives_data = value_biases_derivatives_data + head_index * hidden_depth;

        const TensorMap<Tensor<type, 2>> head_query_weights(head_query_weights_data, depth, hidden_depth);
        const TensorMap<Tensor<type, 2>> head_key_weights(head_key_weights_data, depth, hidden_depth);
        const TensorMap<Tensor<type, 2>> head_value_weights(head_value_weights_data, depth, hidden_depth);

        const TensorMap<Tensor<type, 3>> head_query(head_query_data, input_size, hidden_depth, batch_samples_number);
        const TensorMap<Tensor<type, 3>> head_key(head_key_data, context_size, hidden_depth, batch_samples_number);
        const TensorMap<Tensor<type, 3>> head_value(head_value_data, context_size, hidden_depth, batch_samples_number);

        const TensorMap<Tensor<type, 2>> head_projection_weights(head_projection_weights_data, hidden_depth, depth);

        const TensorMap<Tensor<type, 3>> head_attention_weights(head_attention_weights_data, context_size, input_size, batch_samples_number);
        const TensorMap<Tensor<type, 3>> head_attention_outputs(head_attention_outputs_data, input_size, hidden_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_projection_weights_derivatives(head_projection_weights_derivatives_data, hidden_depth, depth);

        TensorMap<Tensor<type, 3>> head_attention_scores_derivatives(head_attention_scores_derivatives_data, context_size, input_size, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_attention_weights_derivatives(head_attention_weights_derivatives_data, context_size, input_size, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_attention_output_derivatives(head_attention_output_derivatives_data, input_size, hidden_depth, batch_samples_number);

        TensorMap<Tensor<type, 3>> head_query_derivatives(head_query_derivatives_data, input_size, hidden_depth, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_key_derivatives(head_key_derivatives_data, context_size, hidden_depth, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_value_derivatives(head_value_derivatives_data, context_size, hidden_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_query_weights_derivatives(head_query_weights_derivatives_data, depth, hidden_depth);
        TensorMap<Tensor<type, 2>> head_key_weights_derivatives(head_key_weights_derivatives_data, depth, hidden_depth);
        TensorMap<Tensor<type, 2>> head_value_weights_derivatives(head_value_weights_derivatives_data, depth, hidden_depth);

        TensorMap<Tensor<type, 1>> head_query_biases_derivatives(head_query_biases_derivatives_data, hidden_depth);
        TensorMap<Tensor<type, 1>> head_key_biases_derivatives(head_key_biases_derivatives_data, hidden_depth);
        TensorMap<Tensor<type, 1>> head_value_biases_derivatives(head_value_biases_derivatives_data, hidden_depth);
        
        // PROJECTION WEIGHTS DERIVATIVES
        
        head_projection_weights_derivatives.device(*thread_pool_device)
            = head_attention_outputs.contract(deltas, projection_weights_derivatives_contraction_indices);
        
        // ATTENTION OUTPUT DERIVATIVES
        
        for(Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            type* sample_attention_output_derivatives_data = head_attention_output_derivatives_data + sample_index * input_size * hidden_depth;

            TensorMap<Tensor<type, 2>> sample_attention_output_derivatives(sample_attention_output_derivatives_data, input_size, hidden_depth);

            sample_deltas = deltas.chip(sample_index, 0);

            sample_attention_output_derivatives.device(*thread_pool_device)
                = sample_deltas.contract(head_projection_weights, A_BT);
        }

        // VALUE DERIVATIVES
        
        batch_matrix_multiplication(thread_pool_device, head_attention_weights, head_attention_output_derivatives, head_value_derivatives, A_B);

        // VALUE WEIGHTS DERIVATIVES

        head_value_weights_derivatives.device(*thread_pool_device)
            = context.contract(head_value_derivatives, transformation_weights_derivatives_contraction_indices);
        
        // ATTENTION WEIGHTS DERIVATIVES
        
        batch_matrix_multiplication(thread_pool_device, head_value, head_attention_output_derivatives, head_attention_weights_derivatives, A_BT);

        // ATTENTION SCORES DERIVATIVES
        
        softmax_derivatives_times_tensor(head_attention_weights, head_attention_weights_derivatives, head_attention_scores_derivatives, aux_rows);

        head_attention_scores_derivatives.device(*thread_pool_device) = head_attention_scores_derivatives * scaling_factor;

        // QUERY DERIVATIVES

        batch_matrix_multiplication(thread_pool_device, head_attention_scores_derivatives, head_key, head_query_derivatives, AT_B);
    
        // KEY DERIVATIVES

        batch_matrix_multiplication(thread_pool_device, head_attention_scores_derivatives, head_query, head_key_derivatives, A_B);

        // QUERY WEIGHTS DERIVATIVES

        head_query_weights_derivatives.device(*thread_pool_device)
            = input.contract(head_query_derivatives, transformation_weights_derivatives_contraction_indices);

        // KEY WEIGHTS DERIVATIVES

        head_key_weights_derivatives.device(*thread_pool_device)
            = context.contract(head_key_derivatives, transformation_weights_derivatives_contraction_indices);

        for(Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            type* sample_query_derivatives_data = head_query_derivatives_data + sample_index * input_size * hidden_depth;
            type* sample_key_derivatives_data = head_key_derivatives_data + sample_index * context_size * hidden_depth;
            type* sample_value_derivatives_data = head_value_derivatives_data + sample_index * context_size * hidden_depth;

            const TensorMap<Tensor<type, 2>> sample_query_derivatives(sample_query_derivatives_data, input_size, hidden_depth);
            const TensorMap<Tensor<type, 2>> sample_key_derivatives(sample_key_derivatives_data, context_size, hidden_depth);
            const TensorMap<Tensor<type, 2>> sample_value_derivatives(sample_value_derivatives_data, context_size, hidden_depth);

            // INPUT DERIVATIVES

            input_derivatives.chip(sample_index, 0).device(*thread_pool_device)
                += sample_query_derivatives.contract(head_query_weights, A_BT);

            // CONTEXT DERIVATIVES

            context_derivatives.chip(sample_index, 0).device(*thread_pool_device) 
                += sample_key_derivatives.contract(head_key_weights, A_BT)
                + sample_value_derivatives.contract(head_value_weights, A_BT);
        }

        // BIASES DERIVATIVES

        head_query_biases_derivatives.device(*thread_pool_device) = head_query_derivatives.sum(biases_derivatives_sum_indices);

        head_key_biases_derivatives.device(*thread_pool_device) = head_key_derivatives.sum(biases_derivatives_sum_indices);
        
        head_value_biases_derivatives.device(*thread_pool_device) = head_value_derivatives.sum(biases_derivatives_sum_indices);
    }

    projection_biases_derivatives.device(*thread_pool_device) = deltas.sum(projection_biases_derivatives_sum_indices);
}


void MultiheadAttentionLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                              const Index& index,
                                              Tensor<type, 1>& gradient) const
{
    MultiheadAttentionLayerBackPropagation* multihead_attention_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation.get());

    const Tensor<type, 3>& query_weights_derivatives = multihead_attention_layer_back_propagation->query_weights_derivatives;
    const Tensor<type, 2>& query_biases_derivatives = multihead_attention_layer_back_propagation->query_biases_derivatives;

    const Tensor<type, 3>& key_weights_derivatives = multihead_attention_layer_back_propagation->key_weights_derivatives;
    const Tensor<type, 2>& key_biases_derivatives = multihead_attention_layer_back_propagation->key_biases_derivatives;

    const Tensor<type, 3>& value_weights_derivatives = multihead_attention_layer_back_propagation->value_weights_derivatives;
    const Tensor<type, 2>& value_biases_derivatives = multihead_attention_layer_back_propagation->value_biases_derivatives;

    const Tensor<type, 3>& projection_weights_derivatives = multihead_attention_layer_back_propagation->projection_weights_derivatives;
    const Tensor<type, 1>& projection_biases_derivatives = multihead_attention_layer_back_propagation->projection_biases_derivatives;

    type* gradient_data = gradient.data();

    Index gradient_index = index;

    memcpy(gradient_data + gradient_index, query_weights_derivatives.data(), query_weights_derivatives.size()*sizeof(type));

    gradient_index += query_weights_derivatives.size();

    memcpy(gradient_data + gradient_index, query_biases_derivatives.data(), query_biases_derivatives.size()*sizeof(type));

    gradient_index += query_biases_derivatives.size();

    memcpy(gradient_data + gradient_index, key_weights_derivatives.data(), key_weights_derivatives.size()*sizeof(type));

    gradient_index += key_weights_derivatives.size();

    memcpy(gradient_data + gradient_index, key_biases_derivatives.data(), key_biases_derivatives.size()*sizeof(type));

    gradient_index += key_biases_derivatives.size();

    memcpy(gradient_data + gradient_index, value_weights_derivatives.data(),value_weights_derivatives.size()*sizeof(type));

    gradient_index += value_weights_derivatives.size();

    memcpy(gradient_data + gradient_index, value_biases_derivatives.data(), value_biases_derivatives.size()*sizeof(type));

    gradient_index += value_biases_derivatives.size();

    memcpy(gradient_data + gradient_index, projection_weights_derivatives.data(), projection_weights_derivatives.size()*sizeof(type));

    gradient_index += projection_weights_derivatives.size();

    memcpy(gradient_data + gradient_index, projection_biases_derivatives.data(), projection_biases_derivatives.size()*sizeof(type));
}


void MultiheadAttentionLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Multihead Attention layer

    const tinyxml2::XMLElement* multihead_attention_layer_element = document.FirstChildElement("MultiheadAttentionLayer");

    if(!multihead_attention_layer_element)
        throw runtime_error("MultiheadAttentionLayer element is nullptr.\n");
    
    // Layer name

    const tinyxml2::XMLElement* layer_name_element = multihead_attention_layer_element->FirstChildElement("Name");

    if(!layer_name_element)
        throw runtime_error("LayerName element is nullptr.\n");

    if(layer_name_element->GetText())
        set_name(layer_name_element->GetText());
    
    // Input size

    const tinyxml2::XMLElement* input_size_element = multihead_attention_layer_element->FirstChildElement("InputSize");

    if(!input_size_element)
        throw runtime_error("InputSize element is nullptr.\n");

    if(input_size_element->GetText())
        set_input_size(Index(stoi(input_size_element->GetText())));
    
    // Context size

    const tinyxml2::XMLElement* context_size_element = multihead_attention_layer_element->FirstChildElement("ContextSize");

    if(!context_size_element)
        throw runtime_error("ContextSize element is nullptr.\n");

    if(context_size_element->GetText())
        set_context_size(Index(stoi(context_size_element->GetText())));
    
    // Embedding depth

    const tinyxml2::XMLElement* depth_element = multihead_attention_layer_element->FirstChildElement("Depth");

    if(!depth_element)
        throw runtime_error("Depth element is nullptr.\n");

    if(depth_element->GetText())
        set_depth(Index(stoi(depth_element->GetText())));
    
    // Number of attention heads

    const tinyxml2::XMLElement* heads_number_element = multihead_attention_layer_element->FirstChildElement("HeadsNumber");

    if(!heads_number_element)
        throw runtime_error("HeadsNumber element is nullptr.\n");

    if(heads_number_element->GetText())
        set_heads_number(Index(stoi(heads_number_element->GetText())));
    
    // Use causal mask

    const tinyxml2::XMLElement* causal_mask_element = multihead_attention_layer_element->FirstChildElement("CausalMask");

    if(!causal_mask_element)
        throw runtime_error("CausalMask element is nullptr.\n");

    if(causal_mask_element->GetText())
        set_causal_mask(string(causal_mask_element->GetText()) == "true");
    
    // Parameters

    const tinyxml2::XMLElement* parameters_element = multihead_attention_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
        throw runtime_error("Parameters element is nullptr.\n");

    if(parameters_element->GetText())
        set_parameters(to_type_vector(parameters_element->GetText(), " "));
}


void MultiheadAttentionLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Multihead Attention layer

    file_stream.OpenElement("MultiheadAttentionLayer");

    // Layer name

    file_stream.OpenElement("Name");
    file_stream.PushText(name.c_str());
    file_stream.CloseElement();

    // Input size

    file_stream.OpenElement("InputSize");
    file_stream.PushText(to_string(get_input_size()).c_str());
    file_stream.CloseElement();

    // Context size

    file_stream.OpenElement("ContextSize");
    file_stream.PushText(to_string(get_context_size()).c_str());
    file_stream.CloseElement();

    // Embedding depth

    file_stream.OpenElement("Depth");
    file_stream.PushText(to_string(get_depth()).c_str());
    file_stream.CloseElement();

    // Number of attention heads

    file_stream.OpenElement("HeadsNumber");
    file_stream.PushText(to_string(get_heads_number()).c_str());
    file_stream.CloseElement();

    // Use causal mask

    file_stream.OpenElement("CausalMask");
    file_stream.PushText(to_string(use_causal_mask ? 1 : 0).c_str());
    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");
    file_stream.PushText(tensor_to_string(get_parameters()).c_str());
    file_stream.CloseElement();

    // Multihead Attention layer (end tag)

    file_stream.CloseElement();
}


pair<type*, dimensions> MultiheadAttentionLayerForwardPropagation::get_outputs_pair() const
{
    MultiheadAttentionLayer* multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(layer);

    const Index input_size = multihead_attention_layer->get_input_size();

    const Index depth = multihead_attention_layer->get_depth();

    return { (type*)outputs_data, {{ batch_samples_number, input_size, depth }} };
}


void MultiheadAttentionLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    MultiheadAttentionLayer* multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index input_size = multihead_attention_layer->get_input_size();

    const Index context_size = multihead_attention_layer->get_context_size();

    const Index depth = multihead_attention_layer->get_depth();

    const Index heads_number = multihead_attention_layer->get_heads_number();

    const Index hidden_depth = multihead_attention_layer->get_weights_depth();

    // Outputs

    outputs.resize(batch_samples_number, input_size, depth);

    outputs_data = outputs.data();

    // Rest of quantities

    query.resize(input_size, hidden_depth, batch_samples_number, heads_number);
    key.resize(context_size, hidden_depth, batch_samples_number, heads_number);
    value.resize(context_size, hidden_depth, batch_samples_number, heads_number);

    sample_matrix.resize(input_size, hidden_depth);

    attention_scores.resize(context_size, input_size, batch_samples_number, heads_number);
    attention_weights.resize(context_size, input_size, batch_samples_number, heads_number);
    attention_outputs.resize(input_size, hidden_depth, batch_samples_number, heads_number);

    projection_outputs.resize(batch_samples_number, input_size, depth, heads_number);
}


void MultiheadAttentionLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    MultiheadAttentionLayer* multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index input_size = multihead_attention_layer->get_input_size();
    const Index context_size = multihead_attention_layer->get_context_size();
    const Index depth = multihead_attention_layer->get_depth();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index hidden_depth = multihead_attention_layer->get_weights_depth();

    error_attention_scores_derivatives.resize(context_size, input_size, batch_samples_number, heads_number);
    error_attention_weights_derivatives.resize(context_size, input_size, batch_samples_number, heads_number);
    error_attention_output_derivatives.resize(input_size, hidden_depth, batch_samples_number, heads_number);

    sample_deltas.resize(input_size, depth);

    error_query_derivatives.resize(input_size, hidden_depth, batch_samples_number, heads_number);
    error_key_derivatives.resize(context_size, hidden_depth, batch_samples_number, heads_number);
    error_value_derivatives.resize(context_size, hidden_depth, batch_samples_number, heads_number);

    query_weights_derivatives.resize(depth, hidden_depth, heads_number);
    key_weights_derivatives.resize(depth, hidden_depth, heads_number);
    value_weights_derivatives.resize(depth, hidden_depth, heads_number);

    projection_weights_derivatives.resize(hidden_depth, depth, heads_number);

    query_biases_derivatives.resize(hidden_depth, heads_number);
    key_biases_derivatives.resize(hidden_depth, heads_number);
    value_biases_derivatives.resize(hidden_depth, heads_number);
    projection_biases_derivatives.resize(depth);

    aux_rows.resize(context_size);

    input_derivatives.resize(batch_samples_number, input_size, depth);
    context_derivatives.resize(batch_samples_number, context_size, depth);
}


vector<pair<type*, dimensions>> MultiheadAttentionLayerBackPropagation::get_input_derivative_pairs() const
{
    MultiheadAttentionLayer* multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(layer);

    const Index input_size = multihead_attention_layer->get_input_size();
    const Index context_size = multihead_attention_layer->get_context_size();
    const Index depth = multihead_attention_layer->get_depth();

    return
    {{(type*)(input_derivatives.data()), {batch_samples_number, input_size, depth}},
     {(type*)(context_derivatives.data()), {batch_samples_number, context_size, depth}} };
}


}

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
