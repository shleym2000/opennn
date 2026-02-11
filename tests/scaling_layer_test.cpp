#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/statistics.h"

using namespace opennn;

TEST(Scaling2dTest, DefaultConstructor)
{
    Scaling<2> scaling_layer_2d({0});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{0});
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{0});
}


TEST(Scaling2dTest, GeneralConstructor)
{
    Scaling<2> scaling_layer_2d({1});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_name(), "Scaling2d");
    EXPECT_EQ(scaling_layer_2d.get_descriptives().size(), 1);
}


TEST(Scaling2dTest, ForwardPropagate)
{
    Index inputs_number = 3;
    Index samples_number = 2;
    bool is_training = true;

    Scaling<2> scaling_layer_2d({ inputs_number });

    // Test None
    {
        scaling_layer_2d.set_scalers("None");
        Tensor2 inputs(samples_number, inputs_number);
        inputs.setConstant(type(10));

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        TensorView input_view = { inputs.data(), {{samples_number, inputs_number}} };
        scaling_layer_2d.forward_propagate({ input_view }, fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(0, 0), 10.0, NUMERIC_LIMITS_MIN);
    }

    // Test MinimumMaximum
    {
        inputs_number = 1;
        samples_number = 3;
        scaling_layer_2d.set({inputs_number});
        scaling_layer_2d.set_scalers("MinimumMaximum");
        scaling_layer_2d.set_min_max_range(0, 1);

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({{type(2)},{type(4)},{type(6)}});

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        TensorView input_view = {inputs.data(), {{samples_number, inputs_number}}};
        scaling_layer_2d.forward_propagate({ input_view }, fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(0, 0), type(1.5), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(outputs(2, 0), type(3.5), NUMERIC_LIMITS_MIN);
    }

    // Test MeanStandardDeviation
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({ inputs_number });
        scaling_layer_2d.set_scalers("MeanStandardDeviation");

        vector<Descriptives> custom_descriptives(inputs_number);
        custom_descriptives[0].set(type(-10.0), type(10.0), type(1.0), type(0.5));
        custom_descriptives[1].set(type(-10.0), type(10.0), type(0.5), type(2.0));
        scaling_layer_2d.set_descriptives(custom_descriptives);

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({ {type(0),type(0)}, {type(2),type(2)} });

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        TensorView input_view = { inputs.data(), {{samples_number, inputs_number}} };
        scaling_layer_2d.forward_propagate({ input_view }, fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(0, 0), type(-2.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(outputs(1, 1), type(0.75), NUMERIC_LIMITS_MIN);
    }

    // Test StandardDeviation
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({ inputs_number });
        scaling_layer_2d.set_scalers("StandardDeviation");

        vector<Descriptives> custom_std_des(inputs_number);
        custom_std_des[0].set(type(-1.0), type(1.0), type(0.0), type(2.0));
        custom_std_des[1].set(type(-1.0), type(1.0), type(0.0), type(0.5));
        scaling_layer_2d.set_descriptives(custom_std_des);

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({ {type(0),type(0)}, {type(2),type(2)} });

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        TensorView input_view = { inputs.data(), {{samples_number, inputs_number}} };
        scaling_layer_2d.forward_propagate({ input_view }, fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(1, 0), type(1.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(outputs(1, 1), type(4.0), NUMERIC_LIMITS_MIN);
    }

    // Test Logarithm
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({ inputs_number });
        scaling_layer_2d.set_scalers("Logarithm");

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({ {type(0),type(0)}, {type(2),type(2)} });

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        TensorView input_view = { inputs.data(), {{samples_number, inputs_number}} };
        scaling_layer_2d.forward_propagate({ input_view }, fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(0, 0), type(0.0), 0.001);
        EXPECT_NEAR(outputs(1, 0), type(1.098612), 0.001);
    }

    // Test ImageMinMax
    {
        inputs_number = 2;
        samples_number = 2;
        scaling_layer_2d.set({inputs_number});
        scaling_layer_2d.set_scalers("ImageMinMax");

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({{type(0),type(255)}, {type(100),type(2)}});

        unique_ptr<LayerForwardPropagation> fw_prop = make_unique<ScalingForwardPropagation<2>>(samples_number, &scaling_layer_2d);
        fw_prop->initialize();
        Tensor1 workspace(get_size(fw_prop->get_workspace_views()));
        link(workspace.data(), fw_prop->get_workspace_views());

        TensorView input_view = {inputs.data(), {{samples_number, inputs_number}}};
        scaling_layer_2d.forward_propagate({ input_view }, fw_prop, is_training);

        Tensor2 outputs = tensor_map<2>(fw_prop->get_outputs());
        EXPECT_NEAR(outputs(0, 1), 1.0, 0.001);
        EXPECT_NEAR(outputs(1, 0), 100.0/255.0, 0.001);
    }
}
