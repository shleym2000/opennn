//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "tinyxml2.h"

using namespace tinyxml2;

namespace opennn
{

class Dataset;
class NeuralNetwork;

struct Descriptives;
struct Histogram;
struct Correlation;

class TestingAnalysis
{

public:

    TestingAnalysis(const NeuralNetwork* = nullptr, const Dataset* = nullptr);

    struct GoodnessOfFitAnalysis
    {
        type determination = type(0);

        Tensor1 targets;
        Tensor1 outputs;

        void set(const Tensor1&, const Tensor1&, type);

        void save(const filesystem::path&) const;

        void print() const;
    };


    struct RocAnalysis
    {
        Tensor2 roc_curve;

        type area_under_curve = 0;

        type confidence_limit = 0;

        type optimal_threshold = 0;

        void print() const;
    };


    struct KolmogorovSmirnovResults
    {
        Tensor2 positive_cumulative_gain;

        Tensor2 negative_cumulative_gain;

        Tensor1 maximum_gain;
    };


    struct BinaryClassificationRates
    {
        vector<Index> true_positives_indices;

        vector<Index> false_positives_indices;

        vector<Index> false_negatives_indices;

        vector<Index> true_negatives_indices;
    };

    // Get

    NeuralNetwork* get_neural_network() const;
    Dataset* get_dataset() const;

    bool get_display() const;

    Index get_batch_size();

    // Set

    void set_neural_network(NeuralNetwork*);
    void set_dataset(Dataset*);

    void set_display(bool);

    void set_threads_number(const int&);

    void set_batch_size(const Index);

    // Checking

    void check() const;

    // Error data

    pair<Tensor<type,2>, Tensor<type,2>> get_targets_and_outputs(const string&) const;

    Tensor2 calculate_error() const;

    Tensor3 calculate_error_data() const;
    Tensor2 calculate_percentage_error_data() const;

    vector<Descriptives> calculate_absolute_errors_descriptives() const;
    vector<Descriptives> calculate_absolute_errors_descriptives(const Tensor2&, const Tensor2&) const;

    vector<Descriptives> calculate_percentage_errors_descriptives() const;
    vector<Descriptives> calculate_percentage_errors_descriptives(const Tensor2&, const Tensor2&) const;

    vector<vector<Descriptives>> calculate_error_data_descriptives() const;
    void print_error_data_descriptives() const;

    vector<Histogram> calculate_error_data_histograms(const Index = 10) const;

    Tensor<Tensor<Index, 1>, 1> calculate_maximal_errors(const Index = 10) const;

    Tensor2 calculate_errors() const;
    Tensor1 calculate_errors(const Tensor2&, const Tensor2&) const;
    Tensor1 calculate_errors(const string&) const;

    Tensor2 calculate_binary_classification_errors() const;
    Tensor1 calculate_binary_classification_errors(const string&) const;

    Tensor2 calculate_multiple_classification_errors() const;
    Tensor1 calculate_multiple_classification_errors(const string&) const;

    type calculate_normalized_squared_error(const Tensor2&, const Tensor2&) const;
    type calculate_cross_entropy_error(const Tensor2&, const Tensor2&) const;
    type calculate_cross_entropy_error_3d(const Tensor3&, const Tensor2&) const;
    type calculate_weighted_squared_error(const Tensor2&, const Tensor2&, const Tensor1& = Tensor1()) const;
    type calculate_Minkowski_error(const Tensor2&, const Tensor2&, const type = type(1.5)) const;

    type calculate_masked_accuracy(const Tensor3&, const Tensor2&) const;

    type calculate_determination(const Tensor1&, const Tensor1&) const;

    // Goodness-of-fit analysis

    Tensor<Correlation, 1> linear_correlation() const;
    Tensor<Correlation, 1> linear_correlation(const Tensor2&, const Tensor2&) const;

    void print_linear_correlations() const;

    Tensor<GoodnessOfFitAnalysis, 1> perform_goodness_of_fit_analysis() const;
    void print_goodness_of_fit_analysis() const;

    // Binary classifcation

    Tensor1 calculate_binary_classification_tests(const type = 0.50) const;

    void print_binary_classification_tests() const;

    // Confusion

    Tensor<Index, 2> calculate_confusion_binary_classification(const Tensor2&, const Tensor2&, type) const;
    Tensor<Index, 2> calculate_confusion_multiple_classification(const Tensor2&, const Tensor2&) const;
    vector<Tensor<Index, 2>> calculate_multilabel_confusion(const type) const;
    Tensor<Index, 2> calculate_confusion(const Tensor2&, const Tensor2&, type = 0.50) const;
    Tensor<Index, 2> calculate_confusion(const type = 0.50) const;

    Tensor<Index, 1> calculate_positives_negatives_rate(const Tensor2&, const Tensor2&) const;

    // ROC curve

    RocAnalysis perform_roc_analysis() const;

    Tensor2 calculate_roc_curve(const Tensor2&, const Tensor2&) const;

    type calculate_area_under_curve(const Tensor2&) const;
    type calculate_area_under_curve_confidence_limit(const Tensor2&, const Tensor2&) const;
    type calculate_optimal_threshold(const Tensor2&) const;

    // Lift Chart

    Tensor2 perform_cumulative_gain_analysis() const;
    Tensor2 calculate_cumulative_gain(const Tensor2&, const Tensor2&) const;
    Tensor2 calculate_negative_cumulative_gain(const Tensor2&, const Tensor2&)const;

    Tensor2 perform_lift_chart_analysis() const;
    Tensor2 calculate_lift_chart(const Tensor2&) const;

    KolmogorovSmirnovResults perform_Kolmogorov_Smirnov_analysis() const;
    Tensor1 calculate_maximum_gain(const Tensor2&, const Tensor2&) const;

    // Output histogram

    vector<Histogram> calculate_output_histogram(const Tensor2&, Index = 10) const;

    // Binary classification rates

    BinaryClassificationRates calculate_binary_classification_rates(const type = 0.50) const;

    vector<Index> calculate_true_positive_samples(const Tensor2&, const Tensor2&, const vector<Index>&, type) const;
    vector<Index> calculate_false_positive_samples(const Tensor2&, const Tensor2&, const vector<Index>&, type) const;
    vector<Index> calculate_false_negative_samples(const Tensor2&, const Tensor2&, const vector<Index>&, type) const;
    vector<Index> calculate_true_negative_samples(const Tensor2&, const Tensor2&, const vector<Index>&, type) const;

    // Multiple classification tests

    Tensor1 calculate_multiple_classification_precision() const;
    Tensor2 calculate_multiple_classification_tests() const;

    // Multiple classification rates

    Tensor<Tensor<Index,1>, 2> calculate_multiple_classification_rates() const;

    Tensor<Tensor<Index,1>, 2> calculate_multiple_classification_rates(const Tensor2&, const Tensor2&, const vector<Index>&) const;

    Tensor<string, 2> calculate_well_classified_samples(const Tensor2&, const Tensor2&, const vector<string>&) const;

    Tensor<string, 2> calculate_misclassified_samples(const Tensor2&, const Tensor2&, const vector<string>&) const;

    // Save

    void save_confusion(const filesystem::path&) const;

    void save_multiple_classification_tests(const filesystem::path&) const;

    void save_well_classified_samples(const Tensor2&, const Tensor2&, const vector<string>&, const filesystem::path&) const;

    void save_misclassified_samples(const Tensor2&, const Tensor2&, const vector<string>&, const filesystem::path&) const;

    void save_well_classified_samples_statistics(const Tensor2&, const Tensor2&, const vector<string>&, const filesystem::path&) const;

    void save_misclassified_samples_statistics(const Tensor2&, const Tensor2&, const vector<string>&, const filesystem::path&) const;

    void save_well_classified_samples_probability_histogram(const Tensor2&, const Tensor2&, const vector<string>&, const filesystem::path&) const;

    void save_well_classified_samples_probability_histogram(const Tensor<string, 2>&, const filesystem::path&) const;

    void save_misclassified_samples_probability_histogram(const Tensor2&, const Tensor2&, const vector<string>&, const filesystem::path&) const;

    void save_misclassified_samples_probability_histogram(const Tensor<string, 2>&, const filesystem::path&) const;

    // Forecasting

    Tensor<Tensor1, 1> calculate_error_autocorrelation(const Index = 10) const;

    Tensor<Tensor1, 1> calculate_inputs_errors_cross_correlation(const Index = 10) const;

    // Transformer

    pair<type, type> test_transformer() const;

    string test_transformer(const vector<string>& context_string, bool imported_vocabulary) const;

    // Serialization

    void from_XML(const XMLDocument&);

    void to_XML(XMLPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

#ifdef OPENNN_CUDA

    Tensor<Index, 2> calculate_confusion_cuda(const type = 0.50) const;

#endif

private:

    unique_ptr<ThreadPool> thread_pool = nullptr;
    unique_ptr<ThreadPoolDevice> device = nullptr;

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    bool display = true;

    Index batch_size = 0;
};

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
