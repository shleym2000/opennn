//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   T E S T   C L A S S             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "mean_squared_error_test.h"


MeanSquaredErrorTest::MeanSquaredErrorTest() : UnitTesting() 
{
    mean_squared_error.set(&neural_network, &data_set);
}


MeanSquaredErrorTest::~MeanSquaredErrorTest()
{
}


void MeanSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   MeanSquaredError mean_squared_error_1;

   assert_true(!mean_squared_error_1.has_neural_network(), LOG);
   assert_true(!mean_squared_error_1.has_data_set(), LOG);

   // Neural network and data set

   MeanSquaredError mean_squared_error_2(&neural_network, &data_set);

   assert_true(mean_squared_error_2.has_neural_network(), LOG);
   assert_true(mean_squared_error_2.has_data_set(), LOG);
}


void MeanSquaredErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   Tensor<type, 2> data;

   Index batch_samples_number;

   DataSetBatch batch;

   Tensor<type, 1> parameters;

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.set_parameters_constant(0.0);

   data_set.set(1, 1, 1);
   data_set.set_data_constant(0.0);

   batch.set(1, &data_set);

   batch_samples_number = batch.get_samples_number();

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});
   neural_network.set_parameters_constant(0.0);

   data_set.set(1, 1, 1);
   data_set.set_data_constant(0.0);
   data_set.set_training();

   forward_propagation.set(batch_samples_number, &neural_network);

   back_propagation.set(batch_samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   mean_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1,1});
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(1, 2, 1);

   data.resize(1, 3);
   data.setValues({{1, 2, 3}});
   data_set.set_data(data);

   neural_network.set_parameters_constant(1);

   forward_propagation.set(batch_samples_number, &neural_network);

   back_propagation.set(batch_samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   mean_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(abs(back_propagation.error - 1) < 1.0e-3, LOG);

   assert_true(back_propagation.error == 1.0, LOG);

}


void MeanSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   Index samples_number;

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_indices;
   Tensor<Index, 1> target_indices;

   DataSetBatch batch;

   Index inputs_number;
   Index outputs_number;
   Index hidden_neurons;

   RecurrentLayer* recurrent_layer = new RecurrentLayer();

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

   PerceptronLayer* perceptron_layer_1 = new PerceptronLayer();
   PerceptronLayer* perceptron_layer_2 = new PerceptronLayer();

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

   LossIndexBackPropagationLM back_propagation_lm;

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   // Trivial test
   {
       samples_number = 10;
       inputs_number = 1;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_constant(0.0);
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       perceptron_layer_1->set(inputs_number, outputs_number);
       neural_network.add_layer(perceptron_layer_1);

       neural_network.set_parameters_constant(0.0);

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);
       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
       assert_true(all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(),
                          [](type i) { return (i-static_cast<type>(0))<numeric_limits<type>::min(); }), LOG);
   }

   neural_network.set();

   // Test perceptron

   {
       samples_number = 10;
       inputs_number = 3;
       outputs_number = 5;

       const Index neurons_number = 6;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       perceptron_layer_1->set(inputs_number, neurons_number);
       perceptron_layer_2->set(neurons_number, outputs_number);

       neural_network.add_layer(perceptron_layer_1);
       neural_network.add_layer(perceptron_layer_2);

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);
       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test perceptron and binary probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       hidden_neurons = 4;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test perceptron and multiple probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       hidden_neurons = 2;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test lstm
   {
       samples_number = 4;
       inputs_number = 3;
       outputs_number = 2;
       hidden_neurons = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       long_short_term_memory_layer->set(inputs_number, hidden_neurons);

       neural_network.add_layer(long_short_term_memory_layer);

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       long_short_term_memory_layer->set_timesteps(2);

       forward_propagation.set(samples_number, &neural_network);

       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent
   {
       samples_number = 4;
       inputs_number = 2;
       outputs_number = 1;
       hidden_neurons = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);

       neural_network.add_layer(recurrent_layer);

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       recurrent_layer->set_timesteps(2);

       forward_propagation.set(samples_number, &neural_network);

       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and perceptron

   {
       samples_number = 4;
       inputs_number = 2;
       outputs_number = 1;
       hidden_neurons = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);
       perceptron_layer_1->set(hidden_neurons, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(perceptron_layer_1);

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       recurrent_layer->set_timesteps(2);

       forward_propagation.set(samples_number, &neural_network);

       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and binary probabilistic
   {
       samples_number = 4;
       inputs_number = 3;
       hidden_neurons = 4;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);
       probabilistic_layer->set(hidden_neurons, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(probabilistic_layer);

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       recurrent_layer->set_timesteps(2);

       forward_propagation.set(samples_number, &neural_network);

       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and multiple probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       hidden_neurons = 2;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);
       probabilistic_layer->set(hidden_neurons, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(probabilistic_layer);

       neural_network.set_parameters_random();

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       recurrent_layer->set_timesteps(2);

       forward_propagation.set(samples_number, &neural_network);

       back_propagation.set(samples_number, &mean_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test Perceptron LM
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons = 3;
       outputs_number = 4;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Approximation, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &mean_squared_error);
       back_propagation_lm.set(samples_number, &mean_squared_error);

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test probabilistic (binary) LM
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons = 3;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &mean_squared_error);
       back_propagation_lm.set(samples_number, &mean_squared_error);

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test probabilistic (multiple) LM
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons = 3;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       back_propagation.set(samples_number, &mean_squared_error);
       back_propagation_lm.set(samples_number, &mean_squared_error);

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }
}


void MeanSquaredErrorTest::test_calculate_squared_errors()
{
   cout << "test_calculate_squared_errors\n";

   Tensor<Index, 1> hidden_layers_size;

   Index parameters;
   
   DataSetBatch batch(1, &data_set);

   Index batch_samples_number = batch.get_samples_number();

   Tensor<type, 1> squared_errors;

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagationLM back_propagation_lm;

   // Test

   neural_network.set(NeuralNetwork::Approximation, {1,1});
   neural_network.set_parameters_random();

   data_set.set(1, 1, 1);
   data_set.set_data_random();

   forward_propagation.set(batch_samples_number, &neural_network);

   back_propagation_lm.set(batch_samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   mean_squared_error.calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
   squared_errors=back_propagation_lm.squared_errors;

//   Eigen::array<int, 2> vector_times_vector = {Eigen::array<int, 2> ({1,1})};

//   const Tensor<type, 0> product_result = squared_errors.contract(squared_errors, vector_times_vector);

//   assert_true(abs(product_result(0) - error) < 1.0e-3, LOG);
}


void MeanSquaredErrorTest::test_calculate_squared_errors_jacobian()
{
   cout << "test_calculate_squared_errors_jacobian\n";

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_indices;
   Tensor<Index, 1> target_indices;

   DataSetBatch batch;

   Tensor<Index, 1> architecture;

   Index samples_number;
   Index inputs_number;
   Index hidden_neurons_number;
   Index outputs_number;

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

   LossIndexBackPropagationLM back_propagation_lm;

   // Test Perceptron
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 4;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       architecture.resize(3);
       architecture[0] = inputs_number;
       architecture[1] = hidden_neurons_number;
       architecture[2] = outputs_number;

       neural_network.set(NeuralNetwork::Approximation, architecture);

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &mean_squared_error);
       back_propagation_lm.set(samples_number, &mean_squared_error);

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       Tensor<type, 2> numerical_squared_errors_jacobian;

       numerical_squared_errors_jacobian = mean_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, 1.0e-3), LOG);
   }

   // Test probabilistic (binary)

   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &mean_squared_error);
       back_propagation_lm.set(samples_number, &mean_squared_error);

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       Tensor<type, 2> numerical_squared_errors_jacobian;

       numerical_squared_errors_jacobian = mean_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, static_cast<type>(1e-3)), LOG);
   }

   // Test probabilistic (multiple)
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       forward_propagation.set(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &mean_squared_error);
       back_propagation_lm.set(samples_number, &mean_squared_error);

       mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

       Tensor<type, 2> numerical_squared_errors_jacobian;

       numerical_squared_errors_jacobian = mean_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, static_cast<type>(1e-3)), LOG);
   }
}


void MeanSquaredErrorTest::run_test_case()
{
   cout << "Running mean squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Error methods

   test_calculate_error();

   test_calculate_error_gradient();

   // Squared errors methods

   //test_calculate_squared_errors();
   test_calculate_squared_errors_jacobian();

   cout << "End of mean squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lemser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lemser General Public License for more details.

// You should have received a copy of the GNU Lemser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
