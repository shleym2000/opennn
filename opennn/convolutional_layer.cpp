//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "convolutional_layer.h"

namespace opennn
{

Convolutional::Convolutional(const dimensions& new_input_dimensions,
                             const dimensions& new_kernel_dimensions,
                             const string& new_activation_function,
                             const dimensions& new_stride_dimensions,
                             const string& new_convolution_type,
                             const bool& new_batch_normaliztion,
                             const string& new_name) : Layer()
{
    name = "Convolutional";

    set(new_input_dimensions,
        new_kernel_dimensions,
        new_activation_function,
        new_stride_dimensions,
        new_convolution_type,
        new_batch_normaliztion,
        new_name);
}


bool Convolutional::get_batch_normalization() const
{
    return batch_normalization;
}


void Convolutional::preprocess_inputs(const Tensor4& inputs,
                                      Tensor4& preprocessed_inputs) const
{
    preprocessed_inputs = (convolution_type == "Same")
        ? inputs.pad(get_paddings())
        : inputs;}


void Convolutional::calculate_convolutions(const Tensor4& inputs, Tensor4& convolutions) const
{  
    const Index kernels_number = get_kernels_number();

//    const TensorMap2 weights_map = tensor_map<2>(weights);
//    const TensorMap1 biases_map = tensor_map<1>(biases);

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
/*
        const TensorMap3 kernel_weights = tensor_map(weights_map, kernel_index);
        TensorMap3 kernel_convolutions = tensor_map(convolutions, kernel_index);

        kernel_convolutions.device(*device) =
            (inputs.convolve(kernel_weights, array_3( 1, 2, 3)))
                .reshape(kernel_convolutions.dimensions()) + biases_map(kernel_index);
*/
    }
}


void Convolutional::forward_propagate(const vector<TensorView>& input_views,
                                      unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                      const bool& is_training)
{
    const TensorMap4 inputs = tensor_map<4>(input_views[0]);

    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(layer_forward_propagation.get());

    TensorMap4 outputs = tensor_map<4>(this_forward_propagation->outputs);

    Tensor4& preprocessed_inputs = this_forward_propagation->preprocessed_inputs;

    Tensor4& activation_derivatives = this_forward_propagation->activation_derivatives;

    preprocess_inputs(inputs, preprocessed_inputs);
/*
    calculate_convolutions(preprocessed_inputs, outputs);

    if(batch_normalization)
        normalize_batch<4>(
            this_forward_propagation->outputs,
            this_forward_propagation->outputs,
            this_forward_propagation->means,
            this_forward_propagation->standard_deviations,
            running_means,
            running_standard_deviations,
            gammas,
            betas,
            is_training);

    is_training
        ? calculate_activations(activation_function, outputs, activation_derivatives)
        : calculate_activations(activation_function, outputs, empty_4);
*/
}


void Convolutional::back_propagate(const vector<TensorView>& input_views,
                                   const vector<TensorView>& delta_views,
                                   unique_ptr<LayerForwardPropagation>& forward_propagation,
                                   unique_ptr<LayerBackPropagation>& back_propagation) const
{

    // Convolutional layer

    const Index batch_size = back_propagation->batch_size;
    const Index input_height = get_input_height();
    const Index input_width = get_input_width();
    const Index input_channels = get_input_channels();

    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index kernel_channels = get_kernel_channels();
    const Index kernel_size = kernel_height * kernel_width * kernel_channels;

    const TensorMap4 inputs = tensor_map<4>(input_views[0]);
    TensorMap4 deltas = tensor_map<4>(delta_views[0]);

    // Forward propagation

    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(forward_propagation.get());

    Tensor4& preprocessed_inputs = this_forward_propagation->preprocessed_inputs;

    const Tensor4& activation_derivatives = this_forward_propagation->activation_derivatives;

    // Back propagation

    TensorMap4 input_deltas = tensor_map<4>(back_propagation->input_deltas[0]);

    ConvolutionalBackPropagation* convolutional_back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation.get());

    TensorMap1 bias_deltas = tensor_map<1>(convolutional_back_propagation->bias_deltas);

    type* weight_deltas_data = convolutional_back_propagation->weight_deltas.data;

    Tensor4& rotated_weights = convolutional_back_propagation->rotated_weights;

    vector<vector<Tensor2>> precomputed_rotated_slices(kernels_number, vector<Tensor2>(input_channels));
    precomputed_rotated_slices.resize(kernels_number);

    input_deltas.setZero();

    const Index pad_height = (input_height + kernel_height - 1) - get_output_height();
    const Index pad_width = (input_width + kernel_width - 1) - get_output_width();
    const Index pad_top = pad_height / 2;
    const Index pad_bottom = pad_height - pad_top;
    const Index pad_left = pad_width / 2;
    const Index pad_right = pad_width - pad_left;

    const array<pair<Index, Index>, 2> paddings
        = { make_pair(pad_top, pad_bottom), make_pair(pad_left, pad_right) };

    // Inputs (for padding same)

    preprocess_inputs(inputs, preprocessed_inputs);

    deltas.device(*device) = deltas*activation_derivatives;

    bias_deltas.device(*device) = deltas.sum(array<Index, 3>({0, 1, 2}));

#pragma omp parallel for
    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        const TensorMap3 kernel_convolution_deltas = tensor_map_(deltas, kernel_index);

        TensorMap4 kernel_weight_deltas(weight_deltas_data + kernel_index*kernel_size,
                                                        1, kernel_height,kernel_width, kernel_channels);

        kernel_weight_deltas = preprocessed_inputs.convolve(kernel_convolution_deltas, array<Index, 3>({0, 1, 2}));
    }

    // Input derivatives
/*
    rotated_weights.device(*device) = weights.reverse(array<Index, 4>({1, 1, 0, 0}));
*/

#pragma omp parallel for //schedule(static)
    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_rotated_weights = tensor_map(rotated_weights,kernel_index);

        for(Index channel_index = 0; channel_index < input_channels; ++channel_index)
            precomputed_rotated_slices[kernel_index][channel_index] = kernel_rotated_weights.chip(channel_index, 2);
    }

    const array<Index, 2> convolution_dimensions_2d = {0, 1};

    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_convolution_deltas = tensor_map_(deltas, kernel_index);

        #pragma omp parallel for
        for(Index image_index = 0; image_index < batch_size; ++image_index)
        {
            const Tensor2 image_kernel_convolutions_derivatives_padded = kernel_convolution_deltas.chip(image_index, 0).pad(paddings);

            for(Index channel_index = 0; channel_index < input_channels; ++channel_index)
            {
                const Tensor2 convolution_result = image_kernel_convolutions_derivatives_padded
                .convolve(precomputed_rotated_slices[kernel_index][channel_index], convolution_dimensions_2d);

                for(Index h = 0; h < input_height; ++h)
                    for(Index w = 0; w < input_width; ++w)
                        input_deltas(image_index, h, w, channel_index) += convolution_result(h, w);
            }
        }
    }
}


const string& Convolutional::get_activation_function() const
{
    return activation_function;
}


Index Convolutional::get_output_height() const
{
    return (convolution_type == "Same")
        ? (get_input_height() + get_row_stride() - 1) / get_row_stride()
        : (get_input_height() - get_kernel_height()) / get_row_stride() + 1;}


Index Convolutional::get_output_width() const
{
    return (convolution_type == "Same")
        ? (get_input_width() + get_column_stride() - 1) / get_column_stride()
        : (get_input_width() - get_kernel_width()) / get_column_stride() + 1;
}


dimensions Convolutional::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Convolutional::get_output_dimensions() const
{
    return { get_output_height(), get_output_width(), get_kernels_number() };
}


string Convolutional::get_convolution_type() const
{
    return convolution_type;
}


Index Convolutional::get_column_stride() const
{
    return column_stride;
}


Index Convolutional::get_row_stride() const
{
    return row_stride;
}


Index Convolutional::get_kernel_height() const
{
    return weights.dims[0];
}


Index Convolutional::get_kernel_width() const
{
    return weights.dims[1];
}


Index Convolutional::get_kernel_channels() const
{
    return weights.dims[2];
}


Index Convolutional::get_kernels_number() const
{

    return weights.dims[3];
}


Index Convolutional::get_padding_height() const
{
    if (convolution_type == "Valid")
        return 0;

    if (convolution_type == "Same")
    {
        const Index output_height = (get_input_height() + get_row_stride() - 1) / get_row_stride();

        const Index total_padding = (output_height - 1) * get_row_stride() + get_kernel_height() - get_input_height();

        return total_padding / 2;
    }

    throw runtime_error("Unknown convolution type");
}


Index Convolutional::get_padding_width() const
{
    if (convolution_type == "Valid")
        return 0;

    if (convolution_type == "Same")
    {
        const Index output_width = (get_input_width() + get_column_stride() - 1) / get_column_stride();

        const Index total_padding = (output_width - 1) * get_column_stride() + get_kernel_width() - get_input_width();

        return total_padding / 2;
    }

    throw runtime_error("Unknown convolution type");
}


void Convolutional::set(const dimensions& new_input_dimensions,
                        const dimensions& new_kernel_dimensions,
                        const string& new_activation_function,
                        const dimensions& new_stride_dimensions,
                        const string& new_convolution_type,
                        const bool& new_batch_normalization,
                        const string& new_label)
{
    if(new_kernel_dimensions.size() != 4)
        throw runtime_error("Kernel dimensions must be 4");

    if (new_stride_dimensions.size() != 2)
        throw runtime_error("Stride dimensions must be 2");

    if (new_kernel_dimensions[0] > new_input_dimensions[0] || new_kernel_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("kernel dimensions cannot be bigger than input dimensions");

    if (new_kernel_dimensions[2] != new_input_dimensions[2])
        throw runtime_error("kernel_channels must match input_channels dimension");

    if (new_stride_dimensions[0] > new_input_dimensions[0] || new_stride_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("Stride dimensions cannot be bigger than input dimensions");

    if (new_convolution_type == "Same" && (new_kernel_dimensions[0] % 2 == 0 || new_kernel_dimensions[1] % 2 == 0))
        throw runtime_error("Kernel dimensions (height and width) must be odd (3x3,5x5 etc) when using 'Same' padding mode to ensure symmetric padding.");

    input_dimensions = new_input_dimensions;

    const Index kernel_height = new_kernel_dimensions[0];
    const Index kernel_width = new_kernel_dimensions[1];
    const Index kernel_channels = new_kernel_dimensions[2];
    const Index kernels_number = new_kernel_dimensions[3];

//    const Index channels_number = get_input_channels();

    set_row_stride(new_stride_dimensions[0]);
    set_column_stride(new_stride_dimensions[1]);

    set_activation_function(new_activation_function);

    set_convolution_type(new_convolution_type);

    biases.dims = {kernels_number};
    weights.dims = {kernel_height, kernel_width, kernel_channels, kernels_number};

    batch_normalization = new_batch_normalization;

    if (batch_normalization)
    {
        running_means.resize(kernels_number);
        running_standard_deviations.resize(kernels_number);

        gammas.dims = {kernels_number};
        betas.dims = {kernels_number};
    }
    else
    {
        gammas.dims.clear();
        betas.dims.clear();
    }

    set_label(new_label);

#ifdef OPENNN_CUDA

    cudnnCreateTensorDescriptor(&biases_tensor_descriptor);

    cudnnSetTensor4dDescriptor(biases_tensor_descriptor,
                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, kernels_number, 1, 1);

    biases_device.descriptor = biases_tensor_descriptor;

    if (batch_normalization)
    {
        scales_device.set_descriptor({1, kernels_number, 1, 1});
        offsets_device.set_descriptor({1, kernels_number, 1, 1});
    }

    cudnnCreateActivationDescriptor(&activation_descriptor);

    cudnnActivationMode_t activation = CUDNN_ACTIVATION_IDENTITY;

    if(activation_function == "Linear")
        activation = CUDNN_ACTIVATION_IDENTITY;
    else if(activation_function == "Logistic")
        activation = CUDNN_ACTIVATION_SIGMOID;
    else if (activation_function == "HyperbolicTangent")
        activation = CUDNN_ACTIVATION_TANH;
    else if(activation_function == "RectifiedLinear")
        activation = CUDNN_ACTIVATION_RELU;
    else if(activation_function == "ScaledExponentialLinear")
        activation = CUDNN_ACTIVATION_ELU;
    else if (activation_function == "ClippedRelu")
        activation = CUDNN_ACTIVATION_CLIPPED_RELU;
    else if (activation_function == "Swish")
        activation = CUDNN_ACTIVATION_SWISH;

    cudnnSetActivationDescriptor(activation_descriptor, activation, CUDNN_PROPAGATE_NAN, 0.0);
#endif
}


void Convolutional::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Logistic"
    || new_activation_function == "HyperbolicTangent"
    || new_activation_function == "Linear"
    || new_activation_function == "RectifiedLinear"
    || new_activation_function == "ScaledExponentialLinear")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
}


void Convolutional::set_batch_normalization(const bool& new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}


void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    if(new_convolution_type != "Valid" && new_convolution_type != "Same")
        throw runtime_error("Unknown convolution type: " + new_convolution_type + ".\n");

    convolution_type = new_convolution_type;
}


void Convolutional::set_row_stride(const Index& new_stride_row)
{
    if(new_stride_row <= 0)
        throw runtime_error("EXCEPTION: new_stride_row must be a positive number");

    row_stride = new_stride_row;
}


void Convolutional::set_column_stride(const Index& new_stride_column)
{
    if(new_stride_column <= 0)
        throw runtime_error("EXCEPTION: new_stride_column must be a positive number");

    column_stride = new_stride_column;
}


void Convolutional::set_input_dimensions(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 3)
        throw runtime_error("Input new_input_dimensions.size() must be 3");

    input_dimensions = new_input_dimensions;
}


pair<Index, Index> Convolutional::get_padding() const
{
    return { get_padding_height(), get_padding_width() };
}


array<pair<Index, Index>, 4> Convolutional::get_paddings() const
{
    const Index pad_rows = get_padding().first;
    const Index pad_columns = get_padding().second;

    const array<pair<Index, Index>, 4> paddings =
        { make_pair(0, 0),
         make_pair(pad_rows, pad_rows),
         make_pair(pad_columns, pad_columns),
         make_pair(0, 0) };

    return paddings;
}


Index Convolutional::get_input_height() const
{
    return input_dimensions[0];
}


Index Convolutional::get_input_width() const
{
    return input_dimensions[1];
}


vector<TensorView*> Convolutional::get_parameter_views()
{
    vector<TensorView*> views = {&biases, &weights};

    if (batch_normalization)
        views.insert(views.end(), {&gammas, &betas});

    return views;
}


Index Convolutional::get_input_channels() const
{
    return input_dimensions[2];
}


void Convolutional::print() const
{

    cout << "Convolutional layer" << endl
         << "Input dimensions: " << input_dimensions << endl
         << "Output dimensions: " << get_output_dimensions() << endl
         << "Biases dimensions: " << biases.dims << endl
         << "Weights dimensions: " << weights.dims << endl
         << "biases:" << endl;
    //cout << biases << endl;
    cout << "Weights:" << endl;
    //cout << weights << endl;
}


void Convolutional::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Convolutional");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));
    add_xml_element(printer, "KernelsNumber", to_string(get_kernels_number()));
    add_xml_element(printer, "KernelsHeight", to_string(get_kernel_height()));
    add_xml_element(printer, "KernelsWidth", to_string(get_kernel_width()));
    add_xml_element(printer, "KernelsChannels", to_string(get_kernel_channels()));
    add_xml_element(printer, "Activation", activation_function);
    add_xml_element(printer, "StrideDimensions", dimensions_to_string({ get_column_stride(), get_row_stride() }));
    add_xml_element(printer, "Convolution", convolution_type);
    add_xml_element(printer, "BatchNormalization", to_string(batch_normalization));
/*
    if (batch_normalization)
    {
        add_xml_element(printer, "Scales", tensor_to_string<type, 1>(gammas));
        add_xml_element(printer, "Offsets", tensor_to_string<type, 1>(betas));
        add_xml_element(printer, "MovingMeans", tensor_to_string<type, 1>(running_means));
        add_xml_element(printer, "MovingStandardDeviations", tensor_to_string<type, 1>(running_standard_deviations));
    }
*/
    printer.CloseElement();
}


void Convolutional::from_XML(const XMLDocument& document)
{
    const XMLElement* convolutional_layer_element = document.FirstChildElement("Convolutional");

    if(!convolutional_layer_element)
        throw runtime_error("Convolutional layer element is nullptr.\n");

    set_label(read_xml_string(convolutional_layer_element, "Label"));

    set_input_dimensions(string_to_dimensions(read_xml_string(convolutional_layer_element, "InputDimensions")));

    const Index kernel_height = read_xml_index(convolutional_layer_element, "KernelsHeight");
    const Index kernel_width = read_xml_index(convolutional_layer_element, "KernelsWidth");
    const Index kernel_channels = read_xml_index(convolutional_layer_element, "KernelsChannels");
    const Index kernels_number = read_xml_index(convolutional_layer_element, "KernelsNumber");
/*
    biases.resize(kernels_number);
    weights.resize(kernel_height, kernel_width, kernel_channels, kernels_number);
*/
    set_activation_function(read_xml_string(convolutional_layer_element, "Activation"));

    const dimensions stride_dimensions = string_to_dimensions(read_xml_string(convolutional_layer_element, "StrideDimensions"));
    set_column_stride(stride_dimensions[0]);
    set_row_stride(stride_dimensions[1]);

    set_convolution_type(read_xml_string(convolutional_layer_element, "Convolution"));

    bool use_batch_normalization = false;
    const XMLElement* bn_element = convolutional_layer_element->FirstChildElement("BatchNormalization");
    if (bn_element && bn_element->GetText())
        use_batch_normalization = (string(bn_element->GetText()) == "true");

    set_batch_normalization(use_batch_normalization);
/*
    if (batch_normalization)
    {
        gammas.resize(kernels_number);
        betas.resize(kernels_number);
        running_means.resize(kernels_number);
        running_standard_deviations.resize(kernels_number);

        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Scales"), gammas);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Offsets"), betas);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingMeans"), running_means);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingStandardDeviations"), running_standard_deviations);
    }
*/
}


ConvolutionalForwardPropagation::ConvolutionalForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalForwardPropagation::initialize()
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index input_channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    const Index padding_height = convolutional_layer->get_padding_height();
    const Index padding_width = convolutional_layer->get_padding_width();

    preprocessed_inputs.resize(batch_size,
                               input_height + (padding_height*2),
                               input_width + (padding_width*2),
                               input_channels);

    outputs.dims = {batch_size, output_height, output_width, kernels_number};

    means.resize(kernels_number);

    standard_deviations.resize(kernels_number);

    activation_derivatives.resize(batch_size,
                                  output_height,
                                  output_width,
                                  kernels_number);
}


void ConvolutionalForwardPropagation::print() const
{
    cout << "Convolutional layer" << endl
         << "Outputs:" << endl
         << outputs.dims << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


ConvolutionalBackPropagation::ConvolutionalBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalBackPropagation::initialize()
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();
    const Index kernel_channels = convolutional_layer->get_kernel_channels();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    bias_deltas.dims = {kernels_number};

    weight_deltas.dims = {kernels_number, kernel_height, kernel_width, kernel_channels};

    rotated_weights.resize(kernel_height,
                           kernel_width,
                           kernel_channels,
                           kernels_number);

    input_deltas.resize(1);
    input_deltas[0].dims = {batch_size, input_height, input_width, channels};

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        scales_deltas.dims = {kernels_number};
        offsets_deltas.dims = {kernels_number};
    }
}


vector<TensorView*> ConvolutionalBackPropagation::get_workspace_views()
{
    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);

    vector<TensorView*> views = {&bias_deltas, &weight_deltas};

    if (convolutional_layer->get_batch_normalization())
        views.insert(views.end(), {&scales_deltas, &offsets_deltas});

    return views;
}


void ConvolutionalBackPropagation::print() const
{
/*
    cout << "Convolutional layer back propagation" << endl
         << "Biases derivatives:\n" << endl
         << bias_deltas << endl
         << "Synaptic weights derivatives:\n" << endl
         << weight_deltas << endl;
*/
}


#ifdef OPENNN_CUDA

void Convolutional::forward_propagate_cuda(const vector<TensorViewCuda>& inputs_device,
                                           unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                           const bool& is_training)
{
    // Inputs

    const Index height = get_input_height();
    const Index width = get_input_width();
    const Index channels = get_input_channels();

    const float* input_device = inputs_device[0].data;

    // Forward propagation

    const Index batch_size = forward_propagation_cuda->batch_size;

    ConvolutionalForwardPropagationCuda* convolutional_layer_forward_propagation_cuda
        = static_cast<ConvolutionalForwardPropagationCuda*>(forward_propagation_cuda.get());

    Convolutional* convolutional_layer = static_cast<Convolutional*>(convolutional_layer_forward_propagation_cuda->layer);

    float* convolutions = convolutional_layer_forward_propagation_cuda->convolutions.data;
    float* outputs = convolutional_layer_forward_propagation_cuda->outputs.data;

    float* outputs_buffer = use_convolutions() ? convolutions : outputs;

    void* workspace = convolutional_layer_forward_propagation_cuda->workspace;
    const size_t workspace_bytes = convolutional_layer_forward_propagation_cuda->workspace_bytes;

    const cudnnTensorDescriptor_t input_tensor_descriptor = convolutional_layer_forward_propagation_cuda->input_tensor_descriptor;
    const cudnnTensorDescriptor_t output_tensor_descriptor = convolutional_layer_forward_propagation_cuda->outputs.descriptor;

    const cudnnFilterDescriptor_t kernel_descriptor = convolutional_layer_forward_propagation_cuda->kernel_descriptor;

    const cudnnConvolutionDescriptor_t convolution_descriptor = convolutional_layer_forward_propagation_cuda->convolution_descriptor;
    const cudnnConvolutionFwdAlgo_t convolution_algorithm = convolutional_layer_forward_propagation_cuda->convolution_algorithm;

    if (convolutional_layer_forward_propagation_cuda->is_first_layer)
    {
        type* reordered_inputs_device = convolutional_layer_forward_propagation_cuda->reordered_inputs_device;

        reorder_inputs_cuda(input_device, reordered_inputs_device, batch_size, channels, height, width);

        input_device = reordered_inputs_device;
    }

    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle,
                                        &alpha,
                                        input_tensor_descriptor,
                                        input_device,
                                        kernel_descriptor,
                                        weights_device.data,
                                        convolution_descriptor,
                                        convolution_algorithm,
                                        workspace, workspace_bytes,
                                        &beta,
                                        output_tensor_descriptor,
                                        outputs_buffer));

    // Biases

    CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                               &alpha,
                               biases_device.descriptor,
                               biases_device.data,
                               &alpha,
                               output_tensor_descriptor,
                               outputs_buffer));

    // Batch Normalization

    if (batch_normalization && is_training)
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            output_tensor_descriptor,
            outputs_buffer,
            output_tensor_descriptor,
            outputs_buffer,
            scales_device.descriptor,
            scales_device.data,
            offsets_device.data,
            momentum,
            running_means_device,
            running_variances_device,
            CUDNN_BN_MIN_EPSILON,
            convolutional_layer_forward_propagation_cuda->bn_saved_mean,
            convolutional_layer_forward_propagation_cuda->bn_saved_inv_variance));
    else if (batch_normalization && !is_training)
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            output_tensor_descriptor,
            outputs_buffer,
            output_tensor_descriptor,
            outputs_buffer,
            scales_device.descriptor,
            scales_device.data,
            offsets_device.data,
            running_means_device,
            running_variances_device,
            CUDNN_BN_MIN_EPSILON));

    // Activations

    if (convolutional_layer->get_activation_function() != "Linear")
        CHECK_CUDNN(cudnnActivationForward(cudnn_handle,
                                           activation_descriptor,
                                           &alpha,
                                           output_tensor_descriptor,
                                           outputs_buffer,
                                           &beta,
                                           output_tensor_descriptor,
                                           outputs));
    else
        cudaMemcpy(outputs, convolutions, batch_size * get_outputs_number() * sizeof(type), cudaMemcpyDeviceToDevice);
}


void Convolutional::back_propagate_cuda(const vector<TensorViewCuda>& inputs_device,
                                        const vector<TensorViewCuda>& deltas_device,
                                        unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                        unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Forward propagation

    const TensorViewCuda outputs_view = forward_propagation_cuda->outputs;

    ConvolutionalForwardPropagationCuda* convolutional_layer_forward_propagation_cuda
        = static_cast<ConvolutionalForwardPropagationCuda*>(forward_propagation_cuda.get());

    Convolutional* convolutional_layer = static_cast<Convolutional*>(convolutional_layer_forward_propagation_cuda->layer);

    const type* convolutions = convolutional_layer_forward_propagation_cuda->convolutions.data;

    // Back propagation

    type* input_deltas = back_propagation_cuda->input_deltas[0].data;

    ConvolutionalBackPropagationCuda* convolutional_layer_back_propagation_cuda
        = static_cast<ConvolutionalBackPropagationCuda*>(back_propagation_cuda.get());

    void* backward_data_workspace = convolutional_layer_back_propagation_cuda->backward_data_workspace;
    void* backward_filter_workspace = convolutional_layer_back_propagation_cuda->backward_filter_workspace;

    const size_t backward_data_workspace_bytes = convolutional_layer_back_propagation_cuda->backward_data_workspace_bytes;
    const size_t backward_filter_workspace_bytes = convolutional_layer_back_propagation_cuda->backward_filter_workspace_bytes;

    type* weight_deltas_device = convolutional_layer_back_propagation_cuda->weight_deltas_device.data;
    type* bias_deltas_device = convolutional_layer_back_propagation_cuda->bias_deltas_device.data;

    const cudnnTensorDescriptor_t input_tensor_descriptor = convolutional_layer_back_propagation_cuda->input_tensor_descriptor;
    const cudnnTensorDescriptor_t deltas_tensor_descriptor = convolutional_layer_back_propagation_cuda->deltas_tensor_descriptor;

    const cudnnFilterDescriptor_t kernel_descriptor = convolutional_layer_back_propagation_cuda->kernel_descriptor;

    const cudnnConvolutionDescriptor_t convolution_descriptor = convolutional_layer_back_propagation_cuda->convolution_descriptor;

    // Error combinations derivatives

    const string activation_function = convolutional_layer->get_activation_function();

    if (activation_function != "Linear" && use_convolutions() && convolutions != nullptr)
        CHECK_CUDNN(cudnnActivationBackward(cudnn_handle,
                                                       activation_descriptor,
                                                       &alpha,
                                                       deltas_tensor_descriptor,
                                                       outputs_view.data,
                                                       deltas_tensor_descriptor,
                                                       deltas_device[0].data,
                                                       deltas_tensor_descriptor,
                                                       convolutions,
                                                       &beta,
                                                       deltas_tensor_descriptor,
                                                       deltas_device[0].data));

    if (activation_function != "Linear" && !use_convolutions())
        CHECK_CUDNN(cudnnActivationBackward(cudnn_handle,
                                                       activation_descriptor,
                                                       &alpha,
                                                       deltas_tensor_descriptor,
                                                       outputs_view.data,
                                                       deltas_tensor_descriptor,
                                                       deltas_device[0].data,
                                                       deltas_tensor_descriptor,
                                                       outputs_view.data,
                                                       &beta,
                                                       deltas_tensor_descriptor,
                                                       deltas_device[0].data));

    // Batch Normalization

    if (batch_normalization)
        CHECK_CUDNN(cudnnBatchNormalizationBackward(
            cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            &alpha, &alpha,
            outputs_view.descriptor,
            use_convolutions() ? convolutions : outputs_view.data,
            deltas_tensor_descriptor,
            deltas_device[0].data,
            deltas_tensor_descriptor,
            deltas_device[0].data,
            scales_device.descriptor,
            scales_device.data,
            convolutional_layer_back_propagation_cuda->scales_deltas_device.data,
            convolutional_layer_back_propagation_cuda->offsets_deltas_device.data,
            CUDNN_BN_MIN_EPSILON,
            convolutional_layer_forward_propagation_cuda->bn_saved_mean,
            convolutional_layer_forward_propagation_cuda->bn_saved_inv_variance));

    // Convolution backwards for weights derivatives

    cudnnConvolutionBackwardFilter(cudnn_handle,
                                   &alpha,
                                   input_tensor_descriptor,
                                   inputs_device[0].data,
                                   deltas_tensor_descriptor,
                                   deltas_device[0].data,
                                   convolution_descriptor,
                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                   backward_filter_workspace,
                                   backward_filter_workspace_bytes,
                                   &beta,
                                   kernel_descriptor, weight_deltas_device);

    // Biases gradients

    cudnnConvolutionBackwardBias(cudnn_handle,
                                 &alpha,
                                 deltas_tensor_descriptor,
                                 deltas_device[0].data,
                                 &beta,
                                 biases_device.descriptor,
                                 bias_deltas_device);

    // Convolution backwards for input derivatives

    cudnnConvolutionBackwardData(cudnn_handle,
                                 &alpha,
                                 kernel_descriptor,
                                 weights_device.data,
                                 deltas_tensor_descriptor,
                                 deltas_device[0].data,
                                 convolution_descriptor,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                 backward_data_workspace, backward_data_workspace_bytes,
                                 &beta,
                                 input_tensor_descriptor, input_deltas);
}


vector<TensorViewCuda*> Convolutional::get_parameter_views_device()
{
    vector<TensorViewCuda*> views_device = { &biases_device, &weights_device };

    if (batch_normalization)
        views_device.insert(views_device.end(), {&scales_device, &offsets_device});

    return views_device;
}


void Convolutional::allocate_parameters_device()
{
    // @todo, no hace falta?
    const Index C = get_input_channels();
    const Index R = get_kernel_height();
    const Index S = get_kernel_width();
    const Index K = get_kernels_number();

    if (batch_normalization)
    {
        CHECK_CUDA(cudaMalloc(&running_means_device, K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&running_variances_device, K * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(running_means_device, K * sizeof(float));
        //CUDA_MALLOC_AND_REPORT(running_variances_device, K * sizeof(float));
    }
}


void Convolutional::free()
{
    cudaFree(running_means_device);
    running_means_device = nullptr;

    cudaFree(running_variances_device);
    running_variances_device = nullptr;
}


void Convolutional::copy_parameters_device()
{
    /*
    if(!biases_device.data)
        cout << "Biases device pointer is null" << endl;

    if(!weights_device.data)
        cout << "Weights device pointer is null" << endl;

    if(!biases.data())
        cout << "CPU biases data is null" << endl;

    if(!weights.data())
        cout << "CPU weights data is null" << endl;

    CHECK_CUDA(cudaMemcpy(biases_device, biases.data(), biases.size() * sizeof(type), cudaMemcpyHostToDevice));

    const Index kernel_height = weights.dimension(0);
    const Index kernel_width = weights.dimension(1);
    const Index channels = weights.dimension(2);
    const Index kernels_number = weights.dimension(3);

    Tensor4 weights_for_cudnn_layout(kernel_width, kernel_height, channels, kernels_number);

    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index kernel_height_index = 0; kernel_height_index < kernel_height; ++kernel_height_index)
                for(Index kernel_width_index = 0; kernel_width_index < kernel_width; ++kernel_width_index)
                    weights_for_cudnn_layout(kernel_width_index, kernel_height_index, channel_index, kernel_index)
                        = weights(kernel_height_index, kernel_width_index, channel_index, kernel_index);

    CHECK_CUDA(cudaMemcpy(weights_device, weights_for_cudnn_layout.data(), weights_for_cudnn_layout.size() * sizeof(type), cudaMemcpyHostToDevice));

    if (batch_normalization)
    {
        CHECK_CUDA(cudaMemcpy(scales_device, gammas.data(), gammas.size() * sizeof(type), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(offsets_device, betas.data(), betas.size() * sizeof(type), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(running_means_device, running_means.data(), running_means.size() * sizeof(type), cudaMemcpyHostToDevice));
        Tensor1 moving_variances = running_standard_deviations.square();
        CHECK_CUDA(cudaMemcpy(running_variances_device, moving_variances.data(), moving_variances.size() * sizeof(type), cudaMemcpyHostToDevice));
    }
    */
}


ConvolutionalForwardPropagationCuda::ConvolutionalForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalForwardPropagationCuda::initialize()
{
    Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const bool use_convolutions = convolutional_layer->use_convolutions();

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();
    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();

    const Index pad_height = convolutional_layer->get_padding_height();
    const Index pad_width = convolutional_layer->get_padding_width();

    const Index stride_height = convolutional_layer->get_row_stride();
    const Index stride_width = convolutional_layer->get_column_stride();

    string layer_label = convolutional_layer->get_label();

    if(!layer_label.empty() && layer_label.substr(layer_label.length() - 2) == "_1")
        is_first_layer = true;

    if (is_first_layer)
    {
        CHECK_CUDA(cudaMalloc(&reordered_inputs_device, batch_size * input_height * input_width * channels * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(reordered_inputs_device, batch_size * input_height * input_width * channels * sizeof(float));
    }

    // Kernels

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               kernels_number, channels, kernel_height, kernel_width );

    // Inputs

    cudnnCreateTensorDescriptor(&input_tensor_descriptor);

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size, channels, input_height, input_width );

    // Convolution

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    pad_height, pad_width,
                                    stride_height, stride_width,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT );

    // Outputs

    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                          input_tensor_descriptor, kernel_descriptor,
                                          &output_batch_size, &output_channels, &output_height, &output_width );

    cudnnCreateTensorDescriptor(&outputs.descriptor);

    cudnnSetTensor4dDescriptor(outputs.descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               output_batch_size, output_channels, output_height, output_width );

    CHECK_CUDA(cudaMalloc(&outputs.data, output_batch_size * output_height * output_width * output_channels * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(outputs, output_batch_size * output_height * output_width * output_channels * sizeof(float));

    if (use_convolutions())
    {
        CHECK_CUDA(cudaMalloc(&convolutions.data, output_batch_size * output_height * output_width * output_channels * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(convolutions, output_batch_size * output_height * output_width * output_channels * sizeof(float));
    }

    // Workspace

    convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    cudnnGetConvolutionForwardWorkspaceSize(
        convolutional_layer->get_cudnn_handle(),
        input_tensor_descriptor, kernel_descriptor,
        convolution_descriptor, outputs.descriptor,
        convolution_algorithm, &workspace_bytes);

    if (workspace_bytes > 0)
        CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        CHECK_CUDA(cudaMalloc(&bn_saved_mean, kernels_number * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(bn_saved_mean, kernels_number * sizeof(float));
        CHECK_CUDA(cudaMalloc(&bn_saved_inv_variance, kernels_number * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(bn_saved_inv_variance, kernels_number * sizeof(float));
    }
}


void ConvolutionalForwardPropagationCuda::print() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    cout << layer->get_name() + " forward propagation" << endl;

    cout << "Outputs:" << endl;
    cout << matrix_4d_from_device(outputs.data, batch_size, output_dimensions[0], output_dimensions[1], output_dimensions[2]) << endl;
}


void ConvolutionalForwardPropagationCuda::free()
{
    cudaFree(outputs.data);
    cudaFree(convolutions.data);

    cudaFree(workspace);
    workspace = nullptr;

    cudaFree(reordered_inputs_device);
    reordered_inputs_device = nullptr;

    cudaFree(bn_saved_mean);
    batch_means= nullptr;

    cudaFree(bn_saved_inv_variance);
    bn_saved_inv_variance = nullptr;

    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
    cudnnDestroyTensorDescriptor(outputs.descriptor);

    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}


ConvolutionalBackPropagationCuda::ConvolutionalBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalBackPropagationCuda::initialize()
{
    Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();
    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    const Index pad_height = convolutional_layer->get_padding_height();
    const Index pad_width = convolutional_layer->get_padding_width();

    const Index stride_height = convolutional_layer->get_row_stride();
    const Index stride_width = convolutional_layer->get_column_stride();

    const size_t input_size = batch_size * channels * input_height * input_width;
    const size_t kernel_size = kernels_number * channels * kernel_height * kernel_width;

    // Inputs

    CHECK_CUDA(cudaMalloc(&input_deltas[0].data, input_size * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(input_deltas, input_size * sizeof(float));

    cudnnCreateTensorDescriptor(&input_tensor_descriptor);

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               channels,
                               input_height,
                               input_width);

    // Deltas

    cudnnCreateTensorDescriptor(&deltas_tensor_descriptor);

    cudnnSetTensor4dDescriptor(deltas_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               kernels_number,
                               output_height,
                               output_width);

    // Biases

    CHECK_CUDA(cudaMalloc(&bias_deltas_device.data, kernels_number * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(bias_deltas_device, kernels_number * sizeof(float));

    // Kernel descriptor

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               kernels_number,
                               channels,
                               kernel_height,
                               kernel_width);

    // Kernel derivatives

    CHECK_CUDA(cudaMalloc(&weight_deltas_device.data, kernel_size * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(weight_deltas_device, kernel_size * sizeof(float));

    cudnnCreateFilterDescriptor(&weight_deltas_filter_descriptor);

    cudnnSetFilter4dDescriptor(weight_deltas_filter_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               kernels_number,
                               channels,
                               kernel_height,
                               kernel_width);

    // Convolution descriptor

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    pad_height, pad_width,
                                    stride_height, stride_width,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // Workspace

    cudnnGetConvolutionBackwardDataWorkspaceSize(convolutional_layer->get_cudnn_handle(),
                                                 kernel_descriptor,
                                                 deltas_tensor_descriptor, // ??
                                                 convolution_descriptor,
                                                 input_tensor_descriptor,
                                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                                 &backward_data_workspace_bytes);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(convolutional_layer->get_cudnn_handle(),
                                                   input_tensor_descriptor,
                                                   deltas_tensor_descriptor, // ??
                                                   convolution_descriptor,
                                                   weight_deltas_filter_descriptor,
                                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                                   &backward_filter_workspace_bytes);

    // Workspace memory

    CHECK_CUDA(cudaMalloc(&backward_data_workspace, backward_data_workspace_bytes));

    CHECK_CUDA(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        CHECK_CUDA(cudaMalloc(&scales_deltas_device.data, kernels_number * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(scales_deltas_device, kernels_number * sizeof(float));
        CHECK_CUDA(cudaMalloc(&offsets_deltas_device.data, kernels_number * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(offsets_deltas_device, kernels_number * sizeof(float));
    }
}


vector<TensorViewCuda*> ConvolutionalBackPropagationCuda::get_workspace_views_device()
{
    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);
    /*
    vector<TensorViewCuda*> delta_views_device =
    {
        {bias_deltas_device.data, bias_deltas_device.descriptor},
        {weight_deltas_device.data, weight_deltas_device.descriptor}
    };

    if (convolutional_layer->get_batch_normalization())
    {
        delta_views_device.push_back({ scales_deltas_device.data, scales_deltas_device.descriptor });
        delta_views_device.push_back({ offsets_deltas_device.data, offsets_deltas_device.descriptor });
    }
    */
    vector<TensorViewCuda*> views;
    return views;
}


void ConvolutionalBackPropagationCuda::print() const
{
    const dimensions input_dimensions = layer->get_input_dimensions();
    const dimensions output_dimensions = layer->get_output_dimensions();

    cout << layer->get_name() + " back propagation" << endl;

    cout << "bias_deltas_device" << endl;
    //vector_from_device(bias_deltas,);

    cout << "weight_deltas_device" << endl;
    //matrix_from_device(weight_deltas,);

    cout << "inputs derivatives" << endl;
    matrix_4d_from_device(input_deltas[0].data, batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]);
}


void ConvolutionalBackPropagationCuda::free()
{
    cudaFree(input_deltas[0].data);
    cudaFree(bias_deltas_device.data);
    cudaFree(weight_deltas_device.data);
    cudaFree(backward_data_workspace);
    cudaFree(backward_filter_workspace);

    input_deltas[0].data = nullptr;
    bias_deltas_device.data = nullptr;
    weight_deltas_device.data = nullptr;
    backward_data_workspace = nullptr;
    backward_filter_workspace = nullptr;

    cudnnDestroyTensorDescriptor(deltas_tensor_descriptor);
    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyFilterDescriptor(weight_deltas_filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudaFree(scales_deltas_device.data);
    cudaFree(offsets_deltas_device.data);

    scales_deltas_device.data = nullptr;
    offsets_deltas_device.data = nullptr;
}

REGISTER(LayerForwardPropagationCuda, ConvolutionalForwardPropagationCuda, "Convolutional")
REGISTER(LayerBackPropagationCuda, ConvolutionalBackPropagationCuda, "Convolutional")

#endif

REGISTER(Layer, Convolutional, "Convolutional")
REGISTER(LayerForwardPropagation, ConvolutionalForwardPropagation, "Convolutional")
REGISTER(LayerBackPropagation, ConvolutionalBackPropagation, "Convolutional")

}

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
