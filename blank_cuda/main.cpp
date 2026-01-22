//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/opennn.h"

using namespace opennn;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;

        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Dataset

        ImageDataset image_dataset("/mnt/c/Users/davidgonzalez/Documents/mnist_data");

        image_dataset.set_sample_roles("Training");

        // Neural network

        ImageClassificationNetwork image_classification_network(image_dataset.get_dimensions("Input"),
            {1},
            image_dataset.get_dimensions("Target"));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss_index()->set_regularization_method("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs_number(10);
        adam->set_display_period(1);

#ifdef OPENNN_CUDA
    //training_strategy.train_cuda();
        training_strategy.train();
#else
    training_strategy.train();
#endif

        // Testing analysis

        const TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);

        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;

        cout << "Bye!" << endl;

#ifndef OPENNN_CUDA
        cout << "Enable CUDA in pch.h" << endl;
#endif

        cout << "Completed." << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
