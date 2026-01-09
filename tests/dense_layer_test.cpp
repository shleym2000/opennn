#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/layer.h"
#include "../opennn/dense_layer.h"

using namespace opennn;

TEST(Dense2dTest, DefaultConstructor)
{
    Dense2d dense_layer;

    EXPECT_EQ(dense_layer.get_name(), "Dense2d");
    EXPECT_EQ(dense_layer.get_input_dimensions(), dimensions{ 0 });
    EXPECT_EQ(dense_layer.get_output_dimensions(), dimensions{ 0 });
}


TEST(Dense2dTest, GeneralConstructor)
{
    Dense2d dense_layer({10}, {3}, "Linear");
    
    EXPECT_EQ(dense_layer.get_activation_function(), "Linear");
    EXPECT_EQ(dense_layer.get_input_dimensions(), dimensions{ 10 });
    EXPECT_EQ(dense_layer.get_output_dimensions(), dimensions{ 3 });
    EXPECT_EQ(dense_layer.get_parameters_number(), 33);
}


TEST(Dense2dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 3; 
    const Index outputs_number = 4; 
    const bool is_training = true;

    Dense2d dense2d_layer({ inputs_number }, { outputs_number }, "Linear");
    dense2d_layer.set_parameters_random();

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<Dense2dForwardPropagation>(batch_size, &dense2d_layer);

    Tensor<type, 2> inputs(batch_size, inputs_number);
    inputs.setConstant(1.0f);

    TensorView input_view(inputs.data(), dimensions{ batch_size, inputs_number });
    vector<TensorView> input_views = { input_view };

    ASSERT_NO_THROW(
        dense2d_layer.forward_propagate(input_views, forward_propagation, is_training)
    );

    EXPECT_EQ(dense2d_layer.get_name(), "Dense2d");
    EXPECT_EQ(dense2d_layer.get_input_dimensions(), dimensions{inputs_number});
    EXPECT_EQ(dense2d_layer.get_output_dimensions(), dimensions{outputs_number});

    const TensorView output_view = forward_propagation->get_output_view();

    ASSERT_EQ(output_view.rank(), 2) << "Output should be a 2D tensor (batch_size, outputs_number).";
    EXPECT_EQ(output_view.dims[0], batch_size);
    EXPECT_EQ(output_view.dims[1], outputs_number);
}

