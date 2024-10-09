   // Expression

   string write_expression() const;

   string write_expression_python() const;
   string write_expression_c() const;
   string write_expression_api() const;
   string write_expression_javascript() const;



string write_expression() const
{
    const Index layers_number = get_layers_number();

    const vector<unique_ptr<Layer>>& layers = get_layers();
    const Tensor<string, 1> layer_names = get_layer_names();

    Tensor<string, 1> output_namess_vector;
    Tensor<string, 1> inputs_names_vector;
    inputs_names_vector = inputs_name;
    string aux_name;

    for(int i = 0; i < inputs_name.dimension(0); i++)
    {
        if(!inputs_names_vector[i].empty())
        {
            aux_name = inputs_name[i];
            inputs_names_vector(i) = replace_non_allowed_programming_expressions(aux_name);
        }
        else
        {
            inputs_names_vector(i) = "input_" + to_string(i);
        }
    }

    Index layer_neurons_number;

    Tensor<string, 1> scaled_inputs_names(inputs_name.dimension(0));
    Tensor<string, 1> unscaled_output_namess(inputs_name.dimension(0));

    ostringstream buffer;

    for(Index i = 0; i < layers_number; i++)
    {
        if(i == layers_number-1)
        {
            output_namess_vector = output_names;

            for(int j = 0; j < output_names.dimension(0); j++)
            {
                if(!output_namess_vector[j].empty())
                {
                    aux_name = output_names[j];
                    output_namess_vector(j) = replace_non_allowed_programming_expressions(aux_name);
                }
                else
                {
                    output_namess_vector(j) = "output_" + to_string(i);
                }
            }
			
            buffer << layers[i]->write_expression(inputs_names_vector, output_namess_vector) << endl;
        }
        else
        {
            layer_neurons_number = layers[i]->get_neurons_number();
            output_namess_vector.resize(layer_neurons_number);

            for(Index j = 0; j < layer_neurons_number; j++)
            {
                if(layer_names(i) == "scaling_layer")
                {
                    aux_name = inputs_name(j);
                    output_namess_vector(j) = "scaled_" + replace_non_allowed_programming_expressions(aux_name);
                    scaled_inputs_names(j) = output_namess_vector(j);
                }
                else
                {
                    output_namess_vector(j) =  layer_names(i) + "_output_" + to_string(j);
                }
            }
			
            buffer << layers[i]->write_expression(inputs_names_vector, output_namess_vector) << endl;
            inputs_names_vector = output_namess_vector;
            unscaled_output_namess = inputs_names_vector;
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");

    return expression;
}


string write_expression_c() const
{
    string aux;
    ostringstream buffer;
    ostringstream outputs_buffer;

    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();
    Tensor<string, 1> found_tokens;

    int cell_states_counter = 0;
    int hidden_state_counter = 0;
    int LSTM_number = get_long_short_term_memory_layers_number();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "// Artificial Intelligence Techniques SL\t" << endl
           << "// artelnics@artelnics.com\t" << endl
           << "// Your model has been exported to this c file." << endl
           << "// You can manage it with the main method, where you \t" << endl
           << "// can change the values of your inputs. For example:" << endl
           << "// if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl
           << "// to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl
           << "// main program has to look like this:" << endl
           << "// \t" << endl
           << "// int main(){ " << endl
           << "// \t" << "vector<float> inputs(3);"<< endl
           << "// \t" << endl
           << "// \t" << "const float asdas  = 0.3;" << endl
           << "// \t" << "inputs[0] = asdas;"        << endl
           << "// \t" << "const float input2 = 2.5;" << endl
           << "// \t" << "inputs[1] = input2;"       << endl
           << "// \t" << "const float input3 = 1.8;" << endl
           << "// \t" << "inputs[2] = input3;"       << endl
           << "// \t" << ". . ." << endl
           << "// \n" << endl
           << "// Inputs Names:" <<endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << inputs_outputs_buffer(2)[0]
           << "\n" << endl
           << "#include <iostream>" << endl
           << "#include <vector>" << endl
           << "#include <math.h>" << endl
           << "#include <stdio.h>" << endl
           << "\n" << endl
           << "using namespace std;" << endl
           << "\n" << endl;

    string token;
    string expression = write_expression();

    if(model_type == ModelType::AutoAssociation)
    {
    // Delete intermediate calculations

    // sample_autoassociation_distance
    {
        const string word_to_delete = "sample_autoassociation_distance =";

        const size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }

    // sample_autoassociation_variables_distance
    {
        const string word_to_delete = "sample_autoassociation_variables_distance =";

        const size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }
    }

    stringstream ss(expression);

    Tensor<string, 1> tokens;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{')
            break;

        if(token.size() > 1 && token.back() != ';')
            token += ';';

        push_back_string(tokens, token);
    }

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        const string word = get_word_from_token(t);

        if(word.size() > 1 && !find_string_in_tensor(found_tokens, word))
            push_back_string(found_tokens, word);
    }

    const string target_string0("Logistic");
    const string target_string1("ReLU");
    const string target_string4("ExponentialLinear");
    const string target_string5("SELU");
    const string target_string6("HardSigmoid");
    const string target_string7("SoftPlus");
    const string target_string8("SoftSign");

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        const string t = tokens(i);

        const size_t substring_length0 = t.find(target_string0);
        const size_t substring_length1 = t.find(target_string1);
        const size_t substring_length4 = t.find(target_string4);
        const size_t substring_length5 = t.find(target_string5);
        const size_t substring_length6 = t.find(target_string6);
        const size_t substring_length7 = t.find(target_string7);
        const size_t substring_length8 = t.find(target_string8);

        if(substring_length0 < t.size() && substring_length0!=0){ logistic = true; }
        if(substring_length1 < t.size() && substring_length1!=0){ ReLU = true; }
        if(substring_length4 < t.size() && substring_length4!=0){ ExpLinear = true; }
        if(substring_length5 < t.size() && substring_length5!=0){ SExpLinear = true; }
        if(substring_length6 < t.size() && substring_length6!=0){ HSigmoid = true; }
        if(substring_length7 < t.size() && substring_length7!=0){ SoftPlus = true; }
        if(substring_length8 < t.size() && substring_length8!=0){ SoftSign = true; }
    }

    if(logistic)
    {
        buffer << "float Logistic (float x) {" << endl
               << "float z = 1/(1+exp(-x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "float ReLU(float x) {" << endl
               << "float z = max(0, x);" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "float ExponentialLinear(float x) {" << endl
               << "float z;" << endl
               << "float alpha = 1.67326;" << endl
               << "if(x>0){" << endl
               << "z = x;" << endl
               << "}else{" << endl
               << "z = alpha*(exp(x)-1);" << endl
               << "}" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "float SELU(float x) {" << endl
               << "float z;" << endl
               << "float alpha  = 1.67326;" << endl
               << "float lambda = 1.05070;" << endl
               << "if(x > 0){" << endl
               << "z = lambda*x;" << endl
               << "}else{" << endl
               << "z = lambda*alpha*(exp(x)-1);" << endl
               << "}" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "float HardSigmoid(float x) {" << endl
               << "float z = 1/(1+exp(-x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "float SoftPlus(float x) {" << endl
               << "float z = log(1+exp(x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "float SoftSign(float x) {" << endl
               << "float z = x/(1+abs(x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            const string token = found_tokens(i);

            if(token.find("cell_state") == 0)
                cell_states_counter += 1;

            if(token.find("hidden_state") == 0)
                hidden_state_counter += 1;
        }

        buffer << "struct LSTMMemory" << endl
               << "{" << endl
               << "\t" << "int current_combinations_derivatives = 3;" << endl
               << "\t" << "int time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t" << "float hidden_state_" << to_string(i) << " = type(0);" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t" << "float cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "} lstm; \n\n" << endl
               << "vector<float> calculate_outputs(const vector<float>& inputs, LSTMMemory& lstm)" << endl;
    }
    else
    {
        buffer << "vector<float> calculate_outputs(const vector<float>& inputs)" << endl;
    }

    buffer << "{" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        if(inputs[i].empty())
            buffer << "\t" << "const float " << "input_" << to_string(i) << " = " << "inputs[" << to_string(i) << "];" << endl;
        else
            buffer << "\t" << "const float " << inputs[i] << " = " << "inputs[" << to_string(i) << "];" << endl;

    if(LSTM_number>0)
    {
        buffer << "\n\tif(lstm.time_step_counter%lstm.current_combinations_derivatives == 0 ){" << endl
               << "\t\t" << "lstm.time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t\t" << "lstm.hidden_state_" << to_string(i) << " = type(0);" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t" << "lstm.cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "\t}" << endl;
    }

    buffer << endl;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        const string t = tokens(i);

        if(t.size() <= 1)
            outputs_buffer << endl;
        else
            outputs_buffer << "\t" << t << endl;
    }

    const string keyword = "double";

    string outputs_espresion = outputs_buffer.str();

    replace_substring_in_string(found_tokens, outputs_espresion, keyword);

    if(LSTM_number>0)
    {
        replace_all_appearances(outputs_espresion, "(t)", "");
        replace_all_appearances(outputs_espresion, "(t-1)", "");
        replace_all_appearances(outputs_espresion, "double cell_state", "cell_state");
        replace_all_appearances(outputs_espresion, "double hidden_state", "hidden_state");
        replace_all_appearances(outputs_espresion, "cell_state"  , "lstm.cell_state");
        replace_all_appearances(outputs_espresion, "hidden_state", "lstm.hidden_state");
    }

    buffer << outputs_espresion;

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "c");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
        buffer << fixed_outputs(i) << endl;

    buffer << "\t" << "vector<float> out(" << outputs.size() << ");" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        if(outputs[i].empty())
            buffer << "\t" << "out[" << to_string(i) << "] = " << "output" << to_string(i) << ";"<< endl;
        else
            buffer << "\t" << "out[" << to_string(i) << "] = " << outputs[i] << ";" << endl;

    if(LSTM_number)
        buffer << "\n\t" << "lstm.time_step_counter += 1;" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}"  << endl
           << "\n" << endl
           << "int main(){ \n" << endl
           << "\t" << "vector<float> inputs(" << to_string(inputs.size()) << "); \n" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs[i].empty())
            buffer << "\t" << "const float " << "input_" << to_string(i) <<" =" << " /*enter your value here*/; " << endl
                   << "\t" << "inputs[" << to_string(i) << "] = " << "input_" << to_string(i) << ";" << endl;
        else
            buffer << "\t" << "const float " << inputs[i] << " =" << " /*enter your value here*/; " << endl
                   << "\t" << "inputs[" << to_string(i) << "] = " << inputs[i] << ";" << endl;
    }

    buffer << endl;

    if(LSTM_number > 0)
        buffer << "\t"   << "LSTMMemory lstm;" << "\n" << endl
               << "\t"   << "vector<float> outputs(" << outputs.size() <<");" << endl
               << "\n\t" << "outputs = calculate_outputs(inputs, lstm);" << endl;
    else
        buffer << "\t"   << "vector<float> outputs(" << outputs.size() <<");" << endl
               << "\n\t" << "outputs = calculate_outputs(inputs);" << endl;

    buffer << "" << endl
           << "\t" << "printf(\"These are your outputs:\\n\");" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        if(outputs[i].empty())
            buffer << "\t" << "printf( \"output" << to_string(i) << ":" << " %f \\n\", "<< "outputs[" << to_string(i) << "]" << ");" << endl;
        else
            buffer << "\t" << "printf( \""<< output_names[i] << ":" << " %f \\n\", "<< "outputs[" << to_string(i) << "]" << ");" << endl;
    }

    buffer << "\n\t" << "return 0;" << endl
           << "} \n" << endl;

    const string out = buffer.str();
    //replace_all_appearances(out, "double double double", "double");
    //replace_all_appearances(out, "double double", "double");
    return out;
}


string write_expression_api() const
{
    ostringstream buffer;
    Tensor<string, 1> found_tokens;
    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_states_counter = 0;
    int hidden_state_counter = 0;

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "<!DOCTYPE html>" << endl
           << "<!--" << endl
           << "Artificial Intelligence Techniques SL\t" << endl
           << "artelnics@artelnics.com\t" << endl
           << "" << endl
           << "Your model has been exported to this php file." << endl
           << "You can manage it writting your parameters in the url of your browser.\t" << endl
           << "Example:" << endl
           << "" << endl
           << "\turl = http://localhost/API_example/\t" << endl
           << "\tparameters in the url = http://localhost/API_example/?num=5&num=2&...\t" << endl
           << "\tTo see the ouput refresh the page" << endl
           << "" << endl
           << "\tInputs Names: \t" << endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << inputs_outputs_buffer(2)[0]
        << "" << endl
        << "-->\t" << endl
        << "" << endl
        << "<html lang = \"en\">\n" << endl
        << "<head>\n" << endl
        << "<title>Rest API Client Side Demo</title>\n " << endl
        << "<meta charset = \"utf-8\">" << endl
        << "<meta name = \"viewport\" content = \"width=device-width, initial-scale=1\">" << endl
        << "<link rel = \"stylesheet\" href = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css\">" << endl
        << "<script src = \"https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js\"></script>" << endl
        << "<script src = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js\"></script>" << endl
        << "</head>" << endl
        << "<style>" << endl
        << ".btn{" << endl
        << "background-color: #7393B3" << endl // Gray
        << "border: none;" << endl
        << "color: white;" << endl
        << "padding: 15px 32px;" << endl
        << "text-align: center;" << endl
        << "font-size: 16px;" << endl
        << "}" << endl
        << "</style>" << endl
        << "<body>" << endl
        << "<div class = \"container\">" << endl
        << "<br></br>" << endl
        << "<div class = \"form-group\">" << endl
        << "<p>" << endl
        << "follow the steps defined in the \"index.php\" file" << endl
        << "</p>" << endl
        << "<p>" << endl
        << "Refresh the page to see the prediction" << endl
        << "</p>" << endl
        << "</div>" << endl
        << "<h4>" << endl
        << "<?php" << "\n" << endl;

    string token;
    string expression = write_expression();

    if(model_type == ModelType::AutoAssociation)
    {
    // Delete intermediate calculations

    // sample_autoassociation_distance
    {
        string word_to_delete = "sample_autoassociation_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }

    // sample_autoassociation_variables_distance
    {
        string word_to_delete = "sample_autoassociation_variables_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }
    }

    stringstream ss(expression);
    Tensor<string, 1> tokens;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{') break;
        if(token.size() > 1 && token.back() != ';') token += ';';

        if(token.size() < 2) continue;

        push_back_string(tokens, token);
    }

    string word;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);
        word = get_word_from_token(t);

        if(word.size() > 1)
            push_back_string(found_tokens, word);
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            const string t = found_tokens(i);

            if(token.find("cell_state") == 0)
                cell_states_counter += 1;

            if(token.find("hidden_state") == 0)
                hidden_state_counter += 1;
        }

        buffer << "class NeuralNetwork{" << endl
               << "public $time_steps = 3;" << endl
               << "public $time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "public $" << "hidden_state_" << to_string(i) << " = type(0);" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "public $" << "cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "}" << endl
               << "$nn = new NeuralNetwork;" << endl;
    }

    buffer << "session_start();" << endl
           << "if(isset($_SESSION['lastpage']) && $_SESSION['lastpage'] == __FILE__) { " << endl
           << "if(isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on') " << endl
           << "\t$url = \"https://\"; " << endl
           << "else" << endl
           << "\t$url = \"http://\"; " << endl
           << "\n" << endl
           << "$url.= $_SERVER['HTTP_HOST'];" << endl
           << "$url.= $_SERVER['REQUEST_URI'];" << endl
           << "$url_components = parse_url($url);" << endl
           << "parse_str($url_components['query'], $params);" << endl
           << "\n" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        if(inputs[i].empty())
            buffer << "$num"    + to_string(i) << " = " << "$params['num" + to_string(i) << "'];" << endl
                   << "$input_" + to_string(i) << " = intval(" << "$num"  + to_string(i) << ");"  << endl;
        else
            buffer << "$num" + to_string(i) << " = " << "$params['num" + to_string(i) << "'];" << endl
                   << "$" << inputs[i]      << " = intval(" << "$num"  + to_string(i) << ");"  << endl;

    buffer << "if(" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        if(i != inputs.dimension(0)-1)
            buffer << "is_numeric(" << "$" << "num" + to_string(i) << ") &&" << endl;
        else
            buffer << "is_numeric(" << "$" << "num" + to_string(i) << "))" << endl;

    buffer << "{" << endl
           << "$status=200;" << endl
           << "$status_msg = 'valid parameters';" << endl
           << "}" << endl
           << "else" << endl
           << "{" << endl
           << "$status =400;" << endl
           << "$status_msg = 'invalid parameters';" << endl
           << "}"   << endl;

    if(LSTM_number>0)
    {
        buffer << "if($nn->time_step_counter % $nn->current_combinations_derivatives === 0 ){" << endl
               << "$nn->current_combinations_derivatives = 3;" << endl
               << "$nn->time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "$nn->" << "hidden_state_" << to_string(i) << " = type(0);" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "$nn->" << "cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "}" << endl;
    }

    buffer << "\n" << endl;

    string target_string0("Logistic");
    string target_string1("ReLU");
    string target_string4("ExponentialLinear");
    string target_string5("SELU");
    string target_string6("HardSigmoid");
    string target_string7("SoftPlus");
    string target_string8("SoftSign");

    size_t substring_length0;
    size_t substring_length1;
    size_t substring_length2;
    size_t substring_length3;
    size_t substring_length4;
    size_t substring_length5;
    size_t substring_length6;
    size_t substring_length7;
    size_t substring_length8;

    string new_word;

    Tensor<string, 1> found_tokens_and_input_names = concatenate_string_tensors(inputs, found_tokens);
    found_tokens_and_input_names = sort_string_tensor(found_tokens_and_input_names);

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        substring_length0 = t.find(target_string0);
        substring_length1 = t.find(target_string1);
        substring_length4 = t.find(target_string4);
        substring_length5 = t.find(target_string5);
        substring_length6 = t.find(target_string6);
        substring_length7 = t.find(target_string7);
        substring_length8 = t.find(target_string8);

        if(substring_length0 < t.size() && substring_length0!=0){ logistic     = true; }
        if(substring_length1 < t.size() && substring_length1!=0){ ReLU         = true; }
        if(substring_length4 < t.size() && substring_length4!=0){ ExpLinear    = true; }
        if(substring_length5 < t.size() && substring_length5!=0){ SExpLinear   = true; }
        if(substring_length6 < t.size() && substring_length6!=0){ HSigmoid     = true; }
        if(substring_length7 < t.size() && substring_length7!=0){ SoftPlus     = true; }
        if(substring_length8 < t.size() && substring_length8!=0){ SoftSign     = true; }

        for(int i = 0; i < found_tokens_and_input_names.dimension(0); i++)
        {
            new_word.clear();

            new_word = "$" + found_tokens_and_input_names[i];

            replace_all_word_appearances(t, found_tokens_and_input_names[i], new_word);
        }

        if(LSTM_number > 0)
        {
            replace_all_appearances(t, "(t)"     , "");
            replace_all_appearances(t, "(t-1)"   , "");
            replace_all_appearances(t, "hidden_" , "$hidden_");
            replace_all_appearances(t, "cell_"   , "$cell_");
            replace_all_appearances(t, "$hidden_", "$nn->hidden_");
            replace_all_appearances(t, "$cell_"  , "$nn->cell_");
        }

        buffer << t << endl;
    }

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "php");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
        buffer << fixed_outputs(i) << endl;

    buffer << "if($status === 200){" << endl
           << "$response = ['status' => $status,  'status_message' => $status_msg" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << ", '" << outputs(i) << "' => " << "$" << outputs[i] << endl;

    buffer << "];" << endl
           << "}" << endl
           << "else" << endl
           << "{" << endl
           << "$response = ['status' => $status,  'status_message' => $status_msg" << "];" << endl
           << "}" << endl;

    if(LSTM_number>0)
        buffer << "$nn->time_step_counter += 1;" << endl;

    buffer << "\n" << endl
           << "$json_response_pretty = json_encode($response, JSON_PRETTY_PRINT);" << endl
           << "echo nl2br(\"\\n\" . $json_response_pretty . \"\\n\");" << endl
           << "}else{" << endl
           << "echo \"New page\";" << endl
           << "}" << endl
           << "$_SESSION['lastpage'] = __FILE__;" << endl
           << "?>" << endl
           << "\n" << endl;

    if(logistic)
        buffer << "<?php" << endl
               << "function Logistic(int $x) {" << endl
               << "$z = 1/(1+exp(-$x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;

    if(ReLU)
        buffer << "<?php" << endl
               << "function ReLU(int $x) {" << endl
               << "$z = max(0, $x);" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;

    if(ExpLinear)
        buffer << "<?php" << endl
               << "function ExponentialLinear(int $x) {" << endl
               << "$alpha = 1.6732632423543772848170429916717;" << endl
               << "if($x>0){" << endl
               << "$z=$x;" << endl
               << "}else{" << endl
               << "$z=$alpha*(exp($x)-1);" << endl
               << "}" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;

    if(SExpLinear)
        buffer << "<?php" << endl
               << "function SELU(int $x) {" << endl
               << "$alpha  = 1.67326;" << endl
               << "$lambda = 1.05070;" << endl
               << "if($x>0){" << endl
               << "$z=$lambda*$x;" << endl
               << "}else{" << endl
               << "$z=$lambda*$alpha*(exp($x)-1);" << endl
               << "}" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;

    if(HSigmoid)
        buffer << "<?php" << endl
               << "function HardSigmoid(int $x) {" << endl
               << "$z=1/(1+exp(-$x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;

    if(SoftPlus)
        buffer << "<?php" << endl
               << "function SoftPlus(int $x) {" << endl
               << "$z=log(1+exp($x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;

    if(SoftSign)
        buffer << "<?php" << endl
               << "function SoftSign(int $x) {" << endl
               << "$z=$x/(1+abs($x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;

    buffer << "</h4>" << endl
           << "</div>" << endl
           << "</body>" << endl
           << "</html>" << endl;

    string out = buffer.str();

    replace_all_appearances(out, "$$", "$");
    replace_all_appearances(out, "_$", "_");

    return out;
}


string write_expression_javascript() const
{
    Tensor<string, 1> tokens;
    Tensor<string, 1> found_tokens;
    Tensor<string, 1> found_mathematical_expressions;
    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();

    ostringstream buffer_to_fix;

    string token;
    string expression = write_expression();

    const int maximum_output_variable_numbers = 5;

    if(model_type == ModelType::AutoAssociation)
    {
    // Delete intermediate calculations

    // sample_autoassociation_distance
    {
        string word_to_delete = "sample_autoassociation_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }

    // sample_autoassociation_variables_distance
    {
        string word_to_delete = "sample_autoassociation_variables_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }
    }

    stringstream ss(expression);

    int cell_states_counter = 0;
    int hidden_state_counter = 0;
    int LSTM_number = get_long_short_term_memory_layers_number();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer_to_fix << "<!--" << endl
                  << "Artificial Intelligence Techniques SL\t" << endl
                  << "artelnics@artelnics.com\t" << endl
                  << "" << endl
                  << "Your model has been exported to this JavaScript file." << endl
                  << "You can manage it with the main method, where you \t" << endl
                  << "can change the values of your inputs. For example:" << endl
                  << "" << endl
                  << "if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl
                  << "to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl
                  << "main program has to look like this:" << endl
                  << "\t" << endl
                  << "int neuralNetwork(){ " << endl
                  << "\t" << "vector<float> inputs(3);"<< endl
                  << "\t" << endl
                  << "\t" << "const float asdas  = 0.3;" << endl
                  << "\t" << "inputs[0] = asdas;"        << endl
                  << "\t" << "const float input2 = 2.5;" << endl
                  << "\t" << "inputs[1] = input2;"       << endl
                  << "\t" << "const float input3 = 1.8;" << endl
                  << "\t" << "inputs[2] = input3;"       << endl
                  << "\t" << ". . ." << endl
                  << "\n" << endl
                  << "Inputs Names:" <<endl;
     
     Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer_to_fix);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    ostringstream buffer;

    buffer << inputs_outputs_buffer(2)[0]
           << "-->" << endl
           << "\n" << endl
           << "<!DOCTYPE HTML>" << endl
           << "<html lang=\"en\">" << endl
           << "\n" << endl
           << "<head>" << endl
           << "<link href=\"https://www.neuraldesigner.com/assets/css/neuraldesigner.css\" rel=\"stylesheet\" />" << endl
           << "<link href=\"https://www.neuraldesigner.com/images/fav.ico\" rel=\"shortcut icon\" type=\"image/x-icon\" />" << endl
           << "</head>" << endl
           << "\n" << endl
           << "<style>" << endl
           << "" << endl
           << "body {" << endl
           << "display: flex;" << endl
           << "justify-content: center;" << endl
           << "align-items: center;" << endl
           << "min-height: 100vh;" << endl
           << "margin: 0;" << endl
           << "background-color: #f0f0f0;" << endl
           << "font-family: Arial, sans-serif;" << endl
           << "}" << endl
           << "" << endl
           << ".form {" << endl
           << "border-collapse: collapse;" << endl
           << "width: 80%; " << endl
           << "max-width: 600px; " << endl
           << "margin: 0 auto; " << endl
           << "background-color: #fff; " << endl
           << "box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); " << endl
           << "border: 1px solid #777; " << endl
           << "border-radius: 5px; " << endl
           << "}" << endl
           << "" << endl
           << "input[type=\"number\"] {" << endl
           << "width: 60px; " << endl
           << "text-align: center; " << endl
           << "}" << endl
           << "" << endl
           << ".form th," << endl
           << ".form td {" << endl
           << "padding: 10px;" << endl
           << "text-align: center;" << endl
           << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; " << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".btn {" << endl
           << "background-color: #5da9e9;" << endl
           << "border: none;" << endl
           << "color: white;" << endl
           << "text-align: center;" << endl
           << "font-size: 16px;" << endl
           << "margin: 4px;" << endl
           << "cursor: pointer;" << endl
           << "padding: 10px 20px;" << endl
           << "border-radius: 5px;" << endl
           << "transition: background-color 0.3s ease;" << endl
           << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;" << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".btn:hover {" << endl
           << "background-color: #4b92d3; " << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << "input[type=\"range\"]::-webkit-slider-runnable-track {" << endl
           << "background: #5da9e9;" << endl
           << "height: 0.5rem;" << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << "input[type=\"range\"]::-moz-range-track {" << endl
           << "background: #5da9e9;" << endl
           << "height: 0.5rem;" << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".tabla {" << endl
           << "width: 100%;" << endl
           << "padding: 5px;" << endl
           << "margin: 0; " << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".form th {" << endl
           << "background-color: #f2f2f2;" << endl
           << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;" << endl
           << "}" << endl
           << "</style>" << endl
           << "\n" << endl
           << "<body>" << endl
           << "\n" << endl
           << "<section>" << endl
           << "<br/>" << endl
           << "\n" << endl
           << "<div align=\"center\" style=\"display:block;text-align: center;\">" << endl
           << "<!-- MENU OPTIONS HERE  -->" << endl
           << "<form style=\"display: inline-block;margin-left: auto; margin-right: auto;\">" << endl
           << "\n" << endl
           << "<table border=\"1px\" class=\"form\">" << endl
           << "\n" << endl
           << "INPUTS" << endl;

    if(has_scaling_layer_2d())
    {
        const Tensor<Descriptives, 1> inputs_descriptives = get_scaling_layer_2d()->get_descriptives();

        for(int i = 0; i < inputs.dimension(0); i++)
            buffer << "<!-- "<< to_string(i) <<"scaling layer -->" << endl
                   << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << inputs_name[i] << " </td>" << endl
                   << "<td style=\"text-align:center\">" << endl
                   << "<input type=\"range\" id=\"" << inputs[i] << "\" value=\"" << (inputs_descriptives(i).minimum + inputs_descriptives(i).maximum)/2 << "\" min=\"" << inputs_descriptives(i).minimum << "\" max=\"" << inputs_descriptives(i).maximum << "\" step=\"" << (inputs_descriptives(i).maximum - inputs_descriptives(i).minimum)/type(100) << "\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "_text')\" />" << endl
                   << "<input class=\"tabla\" type=\"number\" id=\"" << inputs[i] << "_text\" value=\"" << (inputs_descriptives(i).minimum + inputs_descriptives(i).maximum)/2 << "\" min=\"" << inputs_descriptives(i).minimum << "\" max=\"" << inputs_descriptives(i).maximum << "\" step=\"" << (inputs_descriptives(i).maximum - inputs_descriptives(i).minimum)/type(100) << "\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "')\">" << endl
                   << "</td>" << endl
                   << "</tr>" << endl
                   << "\n" << endl;
    }
    else
    {
        for(int i = 0; i < inputs.dimension(0); i++)
            buffer << "<!-- "<< to_string(i) <<"no scaling layer -->" << endl
                   << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << inputs_name[i] << " </td>" << endl
                   << "<td style=\"text-align:center\">" << endl
                   << "<input type=\"range\" id=\"" << inputs[i] << "\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "_text')\" />" << endl
                   << "<input class=\"tabla\" type=\"number\" id=\"" << inputs[i] << "_text\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "')\">" << endl
                   << "</td>" << endl
                   << "</tr>" << endl
                   << "\n" << endl;
    }

    buffer << "</table>" << endl
           << "</form>" << endl
           << "\n" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "<!-- HIDDEN INPUTS -->" << endl;

        for(int i = 0; i < outputs.dimension(0); i++)
            buffer << "<input type=\"hidden\" id=\"" << outputs[i] << "\" value=\"\">" << endl;

        buffer << "\n" << endl;
    }

    buffer << "<div align=\"center\">" << endl
           << "<!-- BUTTON HERE -->" << endl
           << "<button class=\"btn\" onclick=\"neuralNetwork()\">calculate outputs</button>" << endl
           << "</div>" << endl
           << "\n" << endl
           << "<br/>" << endl
           << "\n" << endl
           << "<table border=\"1px\" class=\"form\">" << endl
           << "OUTPUTS" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "<tr style=\"height:3.5em\">" << endl
               << "<td> Target </td>" << endl
               << "<td>" << endl
               << "<select id=\"category_select\" onchange=\"updateSelectedCategory()\">" << endl;

        for(int i = 0; i < outputs.dimension(0); i++)
            buffer << "<option value=\"" << outputs[i] << "\">" << output_names[i] << "</option>" << endl;

        buffer << "</select>" << endl
               << "</td>" << endl
               << "</tr>" << endl
               << "\n" << endl
               << "<tr style=\"height:3.5em\">" << endl
               << "<td> Value </td>" << endl
               << "<td>" << endl
               << "<input style=\"text-align:right; padding-right:20px;\" id=\"selected_value\" value=\"\" type=\"text\"  disabled/>" << endl
               << "</td>" << endl
               << "</tr>" << endl
               << "\n" << endl;
    }
    else
    {
        for(int i = 0; i < outputs.dimension(0); i++)
            buffer << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << output_names[i] << " </td>" << endl
                   << "<td>" << endl
                   << "<input style=\"text-align:right; padding-right:20px;\" id=\"" << outputs[i] << "\" value=\"\" type=\"text\"  disabled/>" << endl
                   << "</td>" << endl
                   << "</tr>" << endl
                   << "\n" << endl;
    }

    buffer << "</table>" << endl
           << "\n" << endl
           << "</form>" << endl
           << "</div>" << endl
           << "\n" << endl
           << "</section>" << endl
           << "\n" << endl
           << "<script>" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "function updateSelectedCategory() {" << endl
               << "\tvar selectedCategory = document.getElementById(\"category_select\").value;" << endl
               << "\tvar selectedValueElement = document.getElementById(\"selected_value\");" << endl;

        for(int i = 0; i < outputs.dimension(0); i++) 
            buffer << "\tif(selectedCategory === \"" << outputs[i] << "\") {" << endl
                   << "\t\tselectedValueElement.value = document.getElementById(\"" << outputs[i] << "\").value;" << endl
                   << "\t}" << endl;

        buffer << "}" << endl
               << "\n" << endl;
    }

    buffer << "function neuralNetwork()" << endl
           << "{" << endl
           << "\t" << "var inputs = [];" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        buffer << "\t" << "var " << inputs[i] << " =" << " document.getElementById(\"" << inputs[i] << "\").value; " << endl
               << "\t" << "inputs.push(" << inputs[i] << ");" << endl;

    buffer << "\n" << "\t" << "var outputs = calculate_outputs(inputs); " << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << "\t" << "var " << outputs[i] << " = document.getElementById(\"" << outputs[i] << "\");" << endl
               << "\t" << outputs[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
        buffer << "\t" << "updateSelectedCategory();" << endl;
    //else
    //{
    //    for(int i = 0; i < outputs.dimension(0); i++)
    //    {
    //        buffer << "\t" << "var " << outputs[i] << " = document.getElementById(\"" << outputs[i] << "\");" << endl;
    //        buffer << "\t" << outputs[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;
    //    }
    //}

    buffer << "\t" << "update_LSTM();" << endl
           << "}" << "\n" << endl;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{')
            break; 

        if(token.size() > 1 && token.back() != ';')
            token += ';'; 

        push_back_string(tokens, token);
    }

    buffer << "function calculate_outputs(inputs)" << endl
           << "{" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        buffer << "\t" << "var " << inputs[i] << " = " << "+inputs[" << to_string(i) << "];" << endl;

    buffer << endl;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        const string word = get_word_from_token(tokens(i));

        if(word.size() > 1)
            push_back_string(found_tokens, word);
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            token = found_tokens(i);

            if(token.find("cell_state") == 0)
                cell_states_counter += 1;

            if(token.find("hidden_state") == 0)
                hidden_state_counter += 1;
        }

        buffer << "\t" << "if(time_step_counter % current_combinations_derivatives == 0 ){" << endl
               << "\t\t" << "time_step_counter = 1" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t\t" << "hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t" << "cell_states_" << to_string(i) << " = 0" << endl;

        buffer << "\t}\n" << endl;
    }

    string target_string_0("Logistic");
    string target_string_1("ReLU");
    string target_string_4("ExponentialLinear");
    string target_string_5("SELU");
    string target_string_6("HardSigmoid");
    string target_string_7("SoftPlus");
    string target_string_8("SoftSign");

    string sufix = "Math.";

    push_back_string(found_mathematical_expressions, "exp");
    push_back_string(found_mathematical_expressions, "tanh");
    push_back_string(found_mathematical_expressions, "max");
    push_back_string(found_mathematical_expressions, "min");

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        const size_t substring_length_0 = t.find(target_string_0);
        const size_t substring_length_1 = t.find(target_string_1);
        const size_t substring_length_4 = t.find(target_string_4);
        const size_t substring_length_5 = t.find(target_string_5);
        const size_t substring_length_6 = t.find(target_string_6);
        const size_t substring_length_7 = t.find(target_string_7);
        const size_t substring_length_8 = t.find(target_string_8);

        if(substring_length_1 < t.size() && substring_length_1!=0){ ReLU = true; }
        if(substring_length_0 < t.size() && substring_length_0!=0){ logistic = true; }
        if(substring_length_6 < t.size() && substring_length_6!=0){ HSigmoid = true; }
        if(substring_length_7 < t.size() && substring_length_7!=0){ SoftPlus = true; }
        if(substring_length_8 < t.size() && substring_length_8!=0){ SoftSign = true; }
        if(substring_length_4 < t.size() && substring_length_4!=0){ ExpLinear = true; }
        if(substring_length_5 < t.size() && substring_length_5!=0){ SExpLinear = true; }

        for(int i = 0; i < found_mathematical_expressions.dimension(0); i++)
        {
            string key_word = found_mathematical_expressions(i);
            string new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        t.size() <= 1
            ? buffer << "" << endl
            : buffer << "\t" << "var " << t << endl;
    }

    if(LSTM_number>0)
        buffer << "\t" << "time_step_counter += 1" << "\n" << endl;

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "javascript");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
        buffer << fixed_outputs(i) << endl;

    buffer << "\t" << "var out = [];" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << "\t" << "out.push(" << outputs[i] << ");" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}" << "\n" << endl;

    if(LSTM_number > 0)
    {
        buffer << "\t" << "var steps = 3;            " << endl
               << "\t" << "var current_combinations_derivatives = steps;   " << endl
               << "\t" << "var time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t" << "var " << "var hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t" << "var " << "var cell_states_" << to_string(i) << " = 0" << endl;

        buffer << "\n" << endl;
    }

    if(logistic)
        buffer << "function Logistic(x) {" << endl
               << "\tvar z = 1/(1+Math.exp(x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;

    if(ReLU)
        buffer << "function ReLU(x) {" << endl
               << "\tvar z = Math.max(0, x);" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;

    if(ExpLinear)
        buffer << "function ExponentialLinear(x) {" << endl
               << "\tvar alpha = 1.67326;" << endl
               << "\tif(x>0){" << endl
               << "\t\tvar z = x;" << endl
               << "\t}else{" << endl
               << "\t\tvar z = alpha*(Math.exp(x)-1);" << endl
               << "\t}" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;

    if(SExpLinear)
        buffer << "function SELU(x) {" << endl
               << "\tvar alpha  = 1.67326;" << endl
               << "\tvar lambda = 1.05070;" << endl
               << "\tif(x>0){" << endl
               << "\t\tvar z = lambda*x;" << endl
               << "\t}else{" << endl
               << "\t\tvar z = lambda*alpha*(Math.exp(x)-1);" << endl
               << "\t}" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;

    if(HSigmoid)
        buffer << "function HardSigmoid(x) {" << endl
               << "\tvar z=1/(1+Math.exp(-x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;

    if(SoftPlus)
        buffer << "function SoftPlus(int x) {" << endl
               << "\tvar z=log(1+Math.exp(x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;

    if(SoftSign)
        buffer << "function SoftSign(x) {" << endl
               << "\tvar z=x/(1+Math.abs(x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;

    buffer << "function updateTextInput1(val, id)" << endl
           << "{" << endl
           << "\t"<< "document.getElementById(id).value = val;" << endl
           << "}" << endl
           << "\n" << endl
           << "window.onresize = showDiv;" << endl
           << "\n" << endl
           << "</script>" << endl
           << "\n" << endl
           << "<!--script src=\"https://www.neuraldesigner.com/app/htmlparts/footer.js\"></script-->" << endl
           << "\n" << endl
           << "</body>" << endl
           << "\n" << endl
           << "</html>" << endl;

    string out = buffer.str();

    if(LSTM_number>0)
    {
        replace_all_appearances(out, "(t)", "");
        replace_all_appearances(out, "(t-1)", "");
        replace_all_appearances(out, "var cell_state"  , "cell_state");
        replace_all_appearances(out, "var hidden_state", "hidden_state");
    }

    return out;
}


string write_expression_python() const
{
    ostringstream buffer;

    Tensor<string, 1> found_tokens;
    Tensor<string, 1> found_mathematical_expressions;

    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> original_inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();

//    const Index layers_number = get_layers_number();

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_states_counter = 0;
    int hidden_state_counter = 0;

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "\'\'\' " << endl
           << "Artificial Intelligence Techniques SL\t" << endl
           << "artelnics@artelnics.com\t" << endl
           << "" << endl
           << "Your model has been exported to this python file."  << endl
           << "You can manage it with the 'NeuralNetwork' class.\t" << endl
           << "Example:" << endl
           << "" << endl
           << "\tmodel = NeuralNetwork()\t" << endl
           << "\tsample = [input_1, input_2, input_3, input_4, ...]\t" << endl
           << "\toutputs = model.calculate_outputs(sample)" << endl
           << "\n" << endl
           << "Inputs Names: \t" << endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
    {
        inputs(i) = inputs_outputs_buffer(0)(i);
        buffer << "\t" << i << ") " << inputs(i) << endl;
    }

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << "\n" << endl
           << "You can predict with a batch of samples using calculate_batch_output method\t" << endl
           << "IMPORTANT: input batch must be <class 'numpy.ndarray'> type\t" << endl
           << "Example_1:\t" << endl
           << "\tmodel = NeuralNetwork()\t" << endl
           << "\tinput_batch = np.array([[1, 2], [4, 5]])\t" << endl
           << "\toutputs = model.calculate_batch_output(input_batch)" << endl
           << "Example_2:\t" << endl
           << "\tinput_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})\t" << endl
           << "\toutputs = model.calculate_batch_output(input_batch.values)" << endl
           << "\'\'\' " << endl
           << "\n" << endl;

    Tensor<string, 1> tokens;

    string expression = write_expression();
    string token;

    stringstream ss(expression);

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{')
			break;
		
        if(token.size() > 1 && token.back() != ';') 
			token += ';';

        push_back_string(tokens, token);
    }

    const string target_string0("Logistic");
    const string target_string1("ReLU");
    const string target_string4("ExponentialLinear");
    const string target_string5("SELU");
    const string target_string6("HardSigmoid");
    const string target_string7("SoftPlus");
    const string target_string8("SoftSign");

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string word;
        string t = tokens(i);

        const size_t substring_length0 = t.find(target_string0);
        const size_t substring_length1 = t.find(target_string1);
        const size_t substring_length4 = t.find(target_string4);
        const size_t substring_length5 = t.find(target_string5);
        const size_t substring_length6 = t.find(target_string6);
        const size_t substring_length7 = t.find(target_string7);
        const size_t substring_length8 = t.find(target_string8);

        if(substring_length0 < t.size() && substring_length0!=0)
			logistic = true; 
        if(substring_length1 < t.size() && substring_length1!=0)
			ReLU = true; 
        if(substring_length4 < t.size() && substring_length4!=0)
			ExpLinear = true; 
        if(substring_length5 < t.size() && substring_length5!=0)
			SExpLinear = true; 
        if(substring_length6 < t.size() && substring_length6!=0)
			HSigmoid = true; 
        if(substring_length7 < t.size() && substring_length7!=0)
			SoftPlus = true; 
        if(substring_length8 < t.size() && substring_length8!=0)
			SoftSign = true; 

        word = get_word_from_token(t);

        if(word.size() > 1)
            push_back_string(found_tokens, word);
    }

    for(int i = 0; i< found_tokens.dimension(0); i++)
    {
        const string token = found_tokens(i);

        if(token.find("cell_state") == 0)
            cell_states_counter += 1;

        if(token.find("hidden_state") == 0)
            hidden_state_counter += 1;
    }

    buffer << "import numpy as np" << endl
           << "\n" << endl;
/*
    if(model_type == ModelType::AutoAssociation)
    {
        buffer << "def calculate_distances(input, output):" << endl;
        buffer << "\t" << "return (np.linalg.norm(np.array(input)-np.array(output)))/len(input)" << endl;

        buffer << "\n" << endl;

        buffer << "def calculate_variables_distances(input, output):" << endl;
        buffer << "\t" << "length_vector = len(input)" << endl;
        buffer << "\t" << "variables_distances = [None] * length_vector" << endl;
        buffer << "\t" << "for i in range(length_vector):" << endl;
        buffer << "\t\t" << "variables_distances[i] = (np.linalg.norm(np.array(input[i])-np.array(output[i])))" << endl;
        buffer << "\t" << "return variables_distances" << endl;

        buffer << "\n" << endl;
    }
*/
    buffer << "class NeuralNetwork:" << endl;
/*
    if(model_type == ModelType::AutoAssociation)
    {
        buffer << "\t" << "minimum = " << to_string(distances_descriptives.minimum) << endl;
        buffer << "\t" << "first_quartile = " << to_string(auto_associative_distances_box_plot.first_quartile) << endl;
        buffer << "\t" << "median = " << to_string(auto_associative_distances_box_plot.median) << endl;
        buffer << "\t" << "mean = " << to_string(distances_descriptives.mean) << endl;
        buffer << "\t" << "third_quartile = "  << to_string(auto_associative_distances_box_plot.third_quartile) << endl;
        buffer << "\t" << "maximum = " << to_string(distances_descriptives.maximum) << endl;
        buffer << "\t" << "standard_deviation = " << to_string(distances_descriptives.standard_deviation) << endl;
        buffer << "\n" << endl;
    }
*/
    if(LSTM_number > 0)
    {
        buffer << "\t" << "def __init__(self, ts = 1):" << endl
               << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl
               << "\t\t" << "self.current_combinations_derivatives = ts" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t" << "self.cell_states_" << to_string(i) << " = 0" << endl;

        buffer << "\t\t" << "self.time_step_counter = 1" << endl;
    }
    else
    {
        string inputs_list;

        for(int i = 0; i < original_inputs.size();i++)
        {
            inputs_list += "'" + original_inputs(i) + "'";

            if(i < original_inputs.size() - 1)
                inputs_list += ", ";
        }

        buffer << "\t" << "def __init__(self):" << endl
               << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl
               << "\t\t" << "self.inputs_name = [" << inputs_list << "]" << endl;
    }

    buffer << "\n" << endl;

    if(logistic)
        buffer << "\tdef Logistic (x):" << endl
               << "\t\t" << "z = 1/(1+np.exp(-x))" << endl
               << "\t\t" << "return z" << endl
               << "\n" << endl;

    if(ReLU)
        buffer << "\tdef ReLU (x):" << endl
               << "\t\t" << "z = max(0, x)" << endl
               << "\t\t" << "return z" << endl
               << "\n" << endl;

    if(ExpLinear)
        buffer << "\tdef ExponentialLinear (x):" << endl
               << "\t\t"   << "float alpha = 1.67326" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "z = x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "z = alpha*(np.exp(x)-1)" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;

    if(SExpLinear)
        buffer << "\tdef SELU (x):" << endl
               << "\t\t"   << "float alpha = 1.67326" << endl
               << "\t\t"   << "float lambda = 1.05070" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "z = lambda*x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "z = lambda*alpha*(np.exp(x)-1)" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;

    if(HSigmoid)
        buffer << "\tdef HardSigmoid (x):" << endl
               << "\t\t"   <<  "z = 1/(1+np.exp(-x))" << endl
               << "\t\t"   <<  "return z" << endl
               << "\n" << endl;

    if(SoftPlus)
        buffer << "\tdef SoftPlus (x):" << endl
               << "\t\t"   << "z = log(1+np.exp(x))" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;

    if(SoftSign)
        buffer << "\tdef SoftSign (x):" << endl
               << "\t\t"   << "z = x/(1+abs(x))" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;

    buffer << "\t" << "def calculate_outputs(self, inputs):" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        buffer << "\t\t" << inputs[i] << " = " << "inputs[" << to_string(i) << "]" << endl;

    if(LSTM_number > 0)
    {
        buffer << "\n\t\t" << "if(self.time_step_counter % self.current_combinations_derivatives == 0 ):" << endl
               << "\t\t\t" << "self.t = 1" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t\t" << "self.cell_states_" << to_string(i) << " = 0" << endl;
    }

    buffer << endl;

    found_tokens.resize(0);
    push_back_string(found_tokens, "log");
    push_back_string(found_tokens, "exp");
    push_back_string(found_tokens, "tanh");

    push_back_string(found_mathematical_expressions, "Logistic");
    push_back_string(found_mathematical_expressions, "ReLU");
    push_back_string(found_mathematical_expressions, "ExponentialLinear");
    push_back_string(found_mathematical_expressions, "SELU");
    push_back_string(found_mathematical_expressions, "HardSigmoid");
    push_back_string(found_mathematical_expressions, "SoftPlus");
    push_back_string(found_mathematical_expressions, "SoftSign");

    string sufix;
    string new_word;
    string key_word ;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        sufix = "np.";
        new_word = ""; 
        key_word = "";

        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            key_word = found_tokens(i);
            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        sufix = "NeuralNetwork.";
        new_word = ""; 
        key_word = "";

        for(int i = 0; i < found_mathematical_expressions.dimension(0); i++)
        {
            key_word = found_mathematical_expressions(i);
            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        if(LSTM_number>0)
        {
            replace_all_appearances(t, "(t)", "");
            replace_all_appearances(t, "(t-1)", "");
            replace_all_appearances(t, "cell_state", "self.cell_state");
            replace_all_appearances(t, "hidden_state", "self.hidden_state");
        }

        buffer << "\t\t" << t << endl;
    }

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "python");

    if(model_type != ModelType::AutoAssociation)
        for(int i = 0; i < fixed_outputs.dimension(0); i++)
            buffer << "\t\t" << fixed_outputs(i) << endl;

    buffer << "\t\t" << "out = " << "[None]*" << outputs.size() << "\n" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << "\t\t" << "out[" << to_string(i) << "] = " << outputs[i] << endl;

    if(LSTM_number>0)
        buffer << "\n\t\t" << "self.time_step_counter += 1" << endl;

    if(model_type != ModelType::AutoAssociation)
        buffer << "\n\t\t" << "return out;" << endl;
    else
        buffer << "\n\t\t" << "return out, sample_autoassociation_distance, sample_autoassociation_variables_distance;" << endl;

    buffer << "\n" << endl
           << "\t" << "def calculate_batch_output(self, input_batch):" << endl
           << "\t\toutput_batch = [None]*input_batch.shape[0]\n" << endl
           << "\t\tfor i in range(input_batch.shape[0]):\n" << endl;

    if(has_recurrent_layer())
        buffer << "\t\t\tif(i%self.current_combinations_derivatives == 0):\n" << endl
               << "\t\t\t\tself.hidden_states = "+ to_string(get_recurrent_layer()->get_neurons_number())+"*[0]\n" << endl;

    if(has_long_short_term_memory_layer())
        buffer << "\t\t\tif(i%self.current_combinations_derivatives == 0):\n" << endl
               << "\t\t\t\tself.hidden_states = "+ to_string(get_long_short_term_memory_layer()->get_neurons_number())+"*[0]\n" << endl
               << "\t\t\t\tself.cell_states = "+ to_string(get_long_short_term_memory_layer()->get_neurons_number())+"*[0]\n" << endl;

    buffer << "\t\t\tinputs = list(input_batch[i])\n" << endl
           << "\t\t\toutput = self.calculate_outputs(inputs)\n" << endl
           << "\t\t\toutput_batch[i] = output\n"<< endl
           << "\t\treturn output_batch\n"<<endl
           << "def main():" << endl
           << "\n\tinputs = []" << "\n" << endl;

    for(Index i = 0; i < inputs.size(); i++)
        buffer << "\t" << inputs(i) << " = " << "#- ENTER YOUR VALUE HERE -#" << endl
               << "\t" << "inputs.append(" << inputs(i) << ")" << "\n" << endl;

    buffer << "\t" << "nn = NeuralNetwork()" << endl
           << "\t" << "outputs = nn.calculate_outputs(inputs)" << endl
           << "\t" << "print(outputs)" << endl
           << "\n" << "main()" << endl;

    string out = buffer.str();

    replace(out, ";", "");

    return out;
}

string replace_non_allowed_programming_expressions(string& s)
{
    string out;

    if(s[0] == '$')
        out = s;

    replace_all_appearances(s, "fn", "f_n");
    replace_all_appearances(s, "if", "i_f");
    replace_all_appearances(s, "do", "d_o");
    replace_all_appearances(s, "or", "o_r");
    replace_all_appearances(s, "is", "i_s");
    replace_all_appearances(s, "as", "a_s");
    replace_all_appearances(s, "or", "o_r");
    replace_all_appearances(s, "if", "i_f");
    replace_all_appearances(s, "in", "in_");
    replace_all_appearances(s, "del", "del");
    replace_all_appearances(s, "max","ma_x");
    replace_all_appearances(s, "min","mi_n");
    replace_all_appearances(s, "and", "an_d");
    replace_all_appearances(s, "for", "fo_r");
    replace_all_appearances(s, "die", "di_e");
    replace_all_appearances(s, "int", "in_t");
    replace_all_appearances(s, "new", "ne_w");
    replace_all_appearances(s, "use", "us_e");
    replace_all_appearances(s, "var", "va_r");
    replace_all_appearances(s, "try", "tr_y");
    replace_all_appearances(s, "xor", "xo_r");
    replace_all_appearances(s, "def", "de_f");
    replace_all_appearances(s, "for", "fo_r");
    replace_all_appearances(s, "not", "no_t_");
    replace_all_appearances(s, "rise","ri_se");
    replace_all_appearances(s, "byte", "byt_e");
    replace_all_appearances(s, "echo", "ech_o");
    replace_all_appearances(s, "eval", "eva_l");
    replace_all_appearances(s, "pass", "pa_ss");
    replace_all_appearances(s, "form", "for_m");
    replace_all_appearances(s, "else", "el_se");
    replace_all_appearances(s, "with", "w_ith");
    replace_all_appearances(s, "exit", "exi_t");
    replace_all_appearances(s, "auto", "aut_o");
    replace_all_appearances(s, "enum", "enu_m");
    replace_all_appearances(s, "case", "cas_e");
    replace_all_appearances(s, "char", "cha_r");
    replace_all_appearances(s, "void", "voi_d");
    replace_all_appearances(s, "goto", "got_o");
    replace_all_appearances(s, "long", "lon_g");
    replace_all_appearances(s, "else", "els_e");
    replace_all_appearances(s, "goto", "got_o");
    replace_all_appearances(s, "type", "ty_pe");
    replace_all_appearances(s, "self", "se_lf");
    replace_all_appearances(s, "list", "lis_t");
    replace_all_appearances(s, "None", "No_ne");
    replace_all_appearances(s, "elif", "el_if");
    replace_all_appearances(s, "True", "t_rue_");
    replace_all_appearances(s, "super","sup_er");
    replace_all_appearances(s, "endif", "endi_f");
    replace_all_appearances(s, "await", "awai_t");
    replace_all_appearances(s, "catch", "catc_h");
    replace_all_appearances(s, "class", "clas_s");
    replace_all_appearances(s, "clone", "clon_e");
    replace_all_appearances(s, "empty", "empt_y");
    replace_all_appearances(s, "final", "fina_l");
    replace_all_appearances(s, "break", "brea_k");
    replace_all_appearances(s, "while", "whil_e");
    replace_all_appearances(s, "float", "floa_t");
    replace_all_appearances(s, "union", "unio_n");
    replace_all_appearances(s, "short", "shor_t");
    replace_all_appearances(s, "const", "cons_t");
    replace_all_appearances(s, "match", "matc_h");
    replace_all_appearances(s, "isset", "isse_t");
    replace_all_appearances(s, "while", "whil_e");
    replace_all_appearances(s, "yield", "yiel_d");
    replace_all_appearances(s, "False", "Fa_lse");
    replace_all_appearances(s, "unset", "unse_t");
    replace_all_appearances(s, "print", "prin_t");
    replace_all_appearances(s, "trait", "trai_t");
    replace_all_appearances(s, "throw", "thro_w");
    replace_all_appearances(s, "raise", "rai_se");
    replace_all_appearances(s, "while", "wh_ile");
    replace_all_appearances(s, "yield", "yi_eld");
    replace_all_appearances(s, "break", "bre_ak");
    replace_all_appearances(s, "class", "c_lass");
    replace_all_appearances(s, "string","str_ing");
    replace_all_appearances(s, "except", "exc_ept");
    replace_all_appearances(s, "lambda", "lamb_da");
    replace_all_appearances(s, "assert", "asser_t");
    replace_all_appearances(s, "global", "glo_bal");
    replace_all_appearances(s, "elseif", "elsei_f");
    replace_all_appearances(s, "endfor", "endfo_r");
    replace_all_appearances(s, "static", "stati_c");
    replace_all_appearances(s, "switch", "switc_h");
    replace_all_appearances(s, "struct", "struc_t");
    replace_all_appearances(s, "double", "doubl_e");
    replace_all_appearances(s, "sizeof", "sizeo_f");
    replace_all_appearances(s, "extern", "exter_n");
    replace_all_appearances(s, "signed", "signe_d");
    replace_all_appearances(s, "return", "retur_n");
    replace_all_appearances(s, "global", "globa_l");
    replace_all_appearances(s, "public", "publi_c");
    replace_all_appearances(s, "return", "retur_n");
    replace_all_appearances(s, "static", "stati_c");
    replace_all_appearances(s, "switch", "switc_h");
    replace_all_appearances(s, "import", "imp_ort");
    replace_all_appearances(s, "return", "retu_rn");
    replace_all_appearances(s, "boolea", "boole_an");
    replace_all_appearances(s, "import", "includ_e");
    replace_all_appearances(s, "friend", "frie_end");
    replace_all_appearances(s, "foreach", "foreac_h");
    replace_all_appearances(s, "private", "privat_e");
    replace_all_appearances(s, "require", "requir_e");
    replace_all_appearances(s, "typedef", "typede_f");
    replace_all_appearances(s, "_Packed", "_P_acked");
    replace_all_appearances(s, "default", "defaul_t");
    replace_all_appearances(s, "extends", "extend_s");
    replace_all_appearances(s, "finally", "finall_y");
    replace_all_appearances(s, "finally", "final_ly");
    replace_all_appearances(s, "nonlocal", "nonlo_cal");
    replace_all_appearances(s, "continue", "con_tinue");
    replace_all_appearances(s, "continue", "continu_e");
    replace_all_appearances(s, "volatile", "volatil_e");
    replace_all_appearances(s, "unsigned", "unsigne_d");
    replace_all_appearances(s, "abstract", "abstrac_t");
    replace_all_appearances(s, "register", "registe_r");
    replace_all_appearances(s, "endwhile", "endwhil_e");
    replace_all_appearances(s, "function", "functio_n");
    replace_all_appearances(s, "readonly", "readonl_y");
    replace_all_appearances(s, "arguments", "argument_s");
    replace_all_appearances(s, "endswitch", "endswitc_h");
    replace_all_appearances(s, "protected", "protecte_d");
    replace_all_appearances(s, "insteadof", "insteado_f");
    replace_all_appearances(s, "interface", "interfac_e");
    replace_all_appearances(s, "namespace", "namespac_e");
    replace_all_appearances(s, "enddeclare", "enddeclar_e");
    replace_all_appearances(s, "endforeach", "endforeac_h");
    replace_all_appearances(s, "implements", "implement_s");
    replace_all_appearances(s, "instanceof", "instanceo_f");
    replace_all_appearances(s, "include_once", "include_on_ce_");
    replace_all_appearances(s, "require_once", "require_on_ce_");
    replace_all_appearances(s, "__halt_compiler", "__h_a_l_t_c_o_m_p_i_l_e_r_");

    for(char& c: s)
    {
        if(c=='1'){ out+="_one_";   continue;}
        if(c=='2'){ out+="_two_";   continue;}
        if(c=='3'){ out+="_three_"; continue;}
        if(c=='4'){ out+="_four_";  continue;}
        if(c=='5'){ out+="_five_";  continue;}
        if(c=='6'){ out+="_six_";   continue;}
        if(c=='7'){ out+="_seven_"; continue;}
        if(c=='8'){ out+="_eight_"; continue;}
        if(c=='9'){ out+="_nine_";  continue;}
        if(c=='0'){ out+="_zero_";  continue;}

        if(c=='.'){ out+="_dot_";   continue;}
        if(c=='/'){ out+="_div_";   continue;}
        if(c=='*'){ out+="_mul_";   continue;}
        if(c=='+'){ out+="_sum_";   continue;}
        if(c=='-'){ out+="_res_";   continue;}
        if(c=='='){ out+="_equ_";   continue;}
        if(c=='!'){ out+="_not_";   continue;}
        if(c==','){ out+="_colon_"; continue;}
        if(c==';'){ out+="_semic_"; continue;}
        if(c=='\\'){ out+="_slash_";continue;}

        if(c=='&'){ out+="_amprsn_"; continue;}
        if(c=='?'){ out+="_ntrgtn_"; continue;}
        if(c=='<'){ out+="_lower_" ; continue;}
        if(c=='>'){ out+="_higher_"; continue;}

        if(isalnum(c)!=0){ out += c; continue;}
        if(isalnum(c)==0){ out+='_'; continue;}
    }

    return out;
}
