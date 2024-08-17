//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//#include <fstream>
#include <sstream>
#include <iomanip>
//#include <cmath>
//#include <ctime>

#include "inputs_selection.h"

namespace opennn
{

InputsSelection::InputsSelection()
{
    set_default();
}


InputsSelection::InputsSelection(TrainingStrategy* new_training_strategy)
    : training_strategy(new_training_strategy)
{
    set_default();
}


TrainingStrategy* InputsSelection::get_training_strategy() const
{
#ifdef OPENNN_DEBUG

    if(!training_strategy)
        throw runtime_error("Training strategy pointer is nullptr.\n");

#endif

    return training_strategy;
}


bool InputsSelection::has_training_strategy() const
{
    if(training_strategy)
    {
        return true;
    }
    else
    {
        return false;
    }
}


const Index& InputsSelection::get_trials_number() const
{
    return trials_number;
}


const bool& InputsSelection::get_display() const
{
    return display;
}


const type& InputsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


const Index& InputsSelection::get_maximum_iterations_number() const
{
    return maximum_epochs_number;
}


const type& InputsSelection::get_maximum_time() const
{
    return maximum_time;
}


const type& InputsSelection::get_maximum_correlation() const
{
    return maximum_correlation;
}


const type& InputsSelection::get_minimum_correlation() const
{
    return minimum_correlation;
}


void InputsSelection::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;     
}


void InputsSelection::set_default()
{
    trials_number = 1;

    // Stopping criteria

    selection_error_goal = type(0);

    maximum_epochs_number = 1000;

    maximum_correlation = type(1);
    minimum_correlation = type(0);

    maximum_time = type(36000.0);
}


void InputsSelection::set_trials_number(const Index& new_trials_number)
{
    trials_number = new_trials_number;
}


void InputsSelection::set_display(const bool& new_display)
{
    display = new_display;
}


void InputsSelection::set_selection_error_goal(const type& new_selection_error_goal)
{
#ifdef OPENNN_DEBUG

    if(new_selection_error_goal < 0)
        throw runtime_error("Selection loss goal must be greater or equal than 0.\n");

#endif

    selection_error_goal = new_selection_error_goal;
}


void InputsSelection::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
    
}


void InputsSelection::set_maximum_time(const type& new_maximum_time)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_time < 0)
        throw runtime_error("Maximum time must be greater than 0.\n");

#endif

    maximum_time = new_maximum_time;
}


void InputsSelection::set_maximum_correlation(const type& new_maximum_correlation)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_correlation < 0 || new_maximum_correlation > 1)
        throw runtime_error("Maximum correlation must be comprised between 0 and 1.\n");

#endif

    maximum_correlation = new_maximum_correlation;
}


void InputsSelection::set_minimum_correlation(const type& new_minimum_correlation)
{
#ifdef OPENNN_DEBUG

    if(new_minimum_correlation < 0 || new_minimum_correlation > 1)
        throw runtime_error("Minimum correlation must be comprised between 0 and 1.\n");

#endif

    minimum_correlation = new_minimum_correlation;
}


string InputsSelection::write_stopping_condition(const TrainingResults& results) const
{
    return results.write_stopping_condition();
}


void InputsSelection::check() const
{
    ostringstream buffer;

    if(!training_strategy)
        throw runtime_error("Pointer to training strategy is nullptr.\n");

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    // Neural network

    if(!loss_index->has_neural_network())
        throw runtime_error("Pointer to neural network is nullptr.\n");

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(neural_network->is_empty())
        throw runtime_error("Neural network is empty.\n");

    // Data set

    if(!loss_index->has_data_set())
        throw runtime_error("Pointer to data set is nullptr.\n");

    const DataSet* data_set = loss_index->get_data_set();

    const Index selection_samples_number = data_set->get_selection_samples_number();

    if(selection_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
}


Index InputsSelectionResults::get_epochs_number() const
{
    return training_error_history.size();
}


void InputsSelectionResults::set(const Index& maximum_epochs_number)
{
    training_error_history.resize(maximum_epochs_number);
    training_error_history.setConstant(type(-1));

    selection_error_history.resize(maximum_epochs_number);
    selection_error_history.setConstant(type(-1));

    mean_selection_error_history.resize(maximum_epochs_number);
    mean_selection_error_history.setConstant(type(-1));

    mean_training_error_history.resize(maximum_epochs_number);
    mean_training_error_history.setConstant(type(-1));
}

string InputsSelectionResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case InputsSelection::StoppingCondition::MaximumTime:
        return "MaximumTime";

    case InputsSelection::StoppingCondition::SelectionErrorGoal:
        return "SelectionErrorGoal";

    case InputsSelection::StoppingCondition::MaximumInputs:
        return "MaximumInputs";

    case InputsSelection::StoppingCondition::MinimumInputs:
        return "MinimumInputs";

    case InputsSelection::StoppingCondition::MaximumEpochs:
        return "MaximumEpochs";

    case InputsSelection::StoppingCondition::MaximumSelectionFailures:
        return "MaximumSelectionFailures";

    case InputsSelection::StoppingCondition::CorrelationGoal:
        return "CorrelationGoal";
    default:
        return string();
    }
}

void InputsSelectionResults::resize_history(const Index& new_size)
{
    const Tensor<type, 1> old_training_error_history = training_error_history;
    const Tensor<type, 1> old_selection_error_history = selection_error_history;

    const Tensor<type, 1> old_mean_selection_history = mean_selection_error_history;
    const Tensor<type, 1> old_mean_training_history = mean_training_error_history;

    training_error_history.resize(new_size);
    selection_error_history.resize(new_size);
    mean_training_error_history.resize(new_size);
    mean_selection_error_history.resize(new_size);


    for(Index i = 0; i < new_size; i++)
    {
        training_error_history(i) = old_training_error_history(i);
        selection_error_history(i) = old_selection_error_history(i);
        mean_selection_error_history(i) = old_mean_selection_history(i);
        mean_training_error_history(i) = old_mean_training_history(i);
    }
}


string InputsSelection::write_time(const type& time) const
{

#ifdef OPENNN_DEBUG

    if(time > type(3600e5))
        throw runtime_error("Time must be lower than 10e5 seconds.\n");

    if(time < type(0))
        throw runtime_error("Time must be greater than 0.\n");

#endif

    const int hours = int(time) / 3600;
    int seconds = int(time) % 3600;
    const int minutes = seconds / 60;
    seconds = seconds % 60;

    ostringstream elapsed_time;

    elapsed_time << setfill('0') << setw(2) << hours << ":"
                 << setfill('0') << setw(2) << minutes << ":"
                 << setfill('0') << setw(2) << seconds << endl;

    return elapsed_time.str();
}


Index InputsSelection::get_input_index(const Tensor<DataSet::VariableUse, 1>& uses, const Index& inputs_number) const
{
#ifdef OPENNN_DEBUG

    if(uses.size() < inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "const Index get_input_index(const Tensor<DataSet::VariableUse, 1>, const Index) method.\n"
               << "Size of uses vector("<< uses.size() <<") must be greater than " <<  inputs_number << ".\n";

        throw(buffer.str());
    }
#endif

    Index i = 0;

    Index j = 0;

    while(i < uses.size())
    {
        if(uses[i] == DataSet::VariableUse::Input && inputs_number == j)
        {
            break;
        }
        else if(uses[i] == DataSet::VariableUse::Input)
        {
            i++;
            j++;
        }
        else
        {
            i++;
        }
    }
    return i;
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
