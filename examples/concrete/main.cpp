//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "../../opennn/dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/bounding_layer.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/model_selection.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/optimization_algorithm.h"
#include "../../opennn/stochastic_gradient_descent.h"

#include "../../opennn/response_optimization.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN Response Optimization Example: Concrete " << endl;

        const Index neurons_number = 3;
        const type regularization_weight = 0.0001;

        // DataSet

        Dataset dataset("../data/concrete.csv", ",", true, false);

        // tentativo  dataset.set_raw_variable_types(Dataset::RawVariableType::Numeric);

        //dataset.set_raw_variable_role("SLUMP(cm)", "Target");
        //dataset.set_raw_variable_role("FLOW(cm)", "Target");
        dataset.set_variable_role("Compressive Strength (28-day)(Mpa)", "Target");

        dataset.split_samples_random(type(0.8), type(0), type(0.2));

        // Neural Network

        ApproximationNetwork approximation_network(dataset.get_input_shape(), {neurons_number}, dataset.get_target_shape());

        Bounding* bounding_layer = (Bounding*)approximation_network.get_first("Bounding");

        if(bounding_layer)
            bounding_layer->set_bounding_method("NoBounding");

        // Training strategy

        TrainingStrategy training_strategy(&approximation_network, &dataset);
        training_strategy.set_optimization_algorithm("StochasticGradientDescent");

        StochasticGradientDescent* sgd = (StochasticGradientDescent*)training_strategy.get_optimization_algorithm();
        sgd->set_batch_size(32);
        sgd->set_initial_learning_rate(0.01);
        sgd->set_momentum(0.9);
        sgd->set_nesterov(true);
        sgd->set_initial_decay(0.00001);

        training_strategy.get_loss_index()->set_regularization_method("L2");
        training_strategy.get_loss_index()->set_regularization_weight(regularization_weight);

        TrainingResults training_results = training_strategy.train();

        // Testing analysis

        TestingAnalysis testing_analysis(&approximation_network, &dataset);
        testing_analysis.print_goodness_of_fit_analysis();

        // 4. RESPONSE OPTIMIZATION
        ResponseOptimization optimizer(&approximation_network, &dataset);

        // --- EXPERIMENT A: Single Objective (Maximize Strength) ---
        cout << "\n[Experiment A] Maximizing Compressive Strength..." << endl;

        vector<ResponseOptimization::Condition> single_conds(dataset.get_variables_number());

        single_conds[dataset.get_variable_index("Compressive Strength (28-day)(Mpa)")] = {ResponseOptimization::ConditionType::Maximize};

        //optimizer.set_condition("Compressive Strength (28-day)(Mpa)",ResponseOptimization::ConditionType::Maximize);

         cout << "DEBUG: all set before optimizing"  << endl;

        auto [single_res, fs1] = optimizer.perform_response_optimization(single_conds);


            cout << "Optimal Recipe for Max Strength:" << endl;

            auto names_variables = dataset.get_variable_names();
            for(auto& n : names_variables)
                cout << setw(15) << n;

            cout << endl;

            for(Index i=0; i<single_res.dimension(1); ++i)
                cout << setw(15) << single_res(0, i);
            cout << endl;


        // --- EXPERIMENT B: Multi-Objective (The Trade-off) ---
        // Goals: Maximize Slump, Flow, and Strength while MINIMIZING Cement.
        cout << "\n[Experiment B] Multi-Objective (Max Slump/Flow/Strength, Min Cement)..." << endl;

        vector<ResponseOptimization::Condition> multi_conds(dataset.get_variables_number());
        multi_conds[dataset.get_variable_index("SLUMP(cm)")] = {ResponseOptimization::ConditionType::Maximize};
        multi_conds[dataset.get_variable_index("FLOW(cm)")]  = {ResponseOptimization::ConditionType::Maximize};
        multi_conds[dataset.get_variable_index("Compressive Strength (28-day)(Mpa)")] = {ResponseOptimization::ConditionType::Maximize};
        multi_conds[dataset.get_variable_index("Cement")] = {ResponseOptimization::ConditionType::Minimize};

        auto [pareto_res, fs2] = optimizer.perform_response_optimization(multi_conds);

        cout << "Pareto Front (Found " << pareto_res.dimension(0) << " optimal trade-offs):" << endl;
        auto names = dataset.get_variable_names();
        for(auto& n : names) cout << setw(14) << n.substr(0,13);
        cout << endl;

        // Print first 10 points of the Pareto front
        Index rows_to_show = min((Index)10, (Index)pareto_res.dimension(0));
        for(Index i=0; i < rows_to_show; ++i) {
            for(Index j=0; j < pareto_res.dimension(1); ++j) {
                cout << setw(14) << fixed << setprecision(2) << pareto_res(i, j);
            }
            cout << endl;
        }

        cout << "\nExperiment Complete." << endl;

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}
// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
