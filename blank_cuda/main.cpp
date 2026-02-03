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
        
        cout << "OpenNN. Melanoma Cancer CUDA Example." << endl;

#ifdef OPENNN_CUDA

        // Data set

        ImageDataset image_dataset("/mnt/c/Users/davidgonzalez/Documents/melanoma_dataset_bmp");

        image_dataset.split_samples_random(0.8, 0.0, 0.2);

        // Neural network

        ImageClassificationNetwork image_classification_network(
            image_dataset.get_dimensions("Input"),
            { 64,128,32 },
            image_dataset.get_dimensions("Target"));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss_index()->set_regularization_method("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_display_period(1);
        adam->set_batch_size(16);
        adam->set_maximum_epochs_number(5);

        training_strategy.train_cuda();


        // Testing analysis

        cout << "Calculating Binary classification tests..." << endl;
        const TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);
        testing_analysis.print_binary_classification_tests();

#endif

        cout << "Bye!" << endl;
        
        
        
        /*
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Dataset

        ImageDataset image_dataset("/mnt/c/Users/davidgonzalez/Documents/mnist_data_binary");

        // Neural network

        ImageClassificationNetwork image_classification_network(image_dataset.get_dimensions("Input"),
            {4},
            image_dataset.get_dimensions("Target"));

        // Training strategy
        WeightedSquaredError();
        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss_index()->set_regularization_method("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs_number(200);
        adam->set_display_period(10);

#ifdef OPENNN_CUDA
        training_strategy.train_cuda();
        //training_strategy.train();
#else
    training_strategy.train();
#endif
    
    const TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);

    cout << "Calculating confusion...." << endl;
    //const Tensor<Index, 2> confusion_cuda = testing_analysis.calculate_confusion_cuda();
    const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
    //cout << "\nConfusion matrix CUDA:\n" << confusion_cuda << endl;
    cout << "\nConfusion matrix:\n" << confusion << endl;
    */

    
   
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
