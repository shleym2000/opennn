// Artificial Intelligence Techniques SL	
// artelnics@artelnics.com	
// Your model has been exported to this c file.
// You can manage it with the main method, where you 	
// can change the values of your inputs. For example:
// if we want to add these 3 values (0.3, 2.5 and 1.8)
// to our 3 inputs (Input_1, Input_2 and Input_1), the
// main program has to look like this:
// 	
// int main(){ 
// 	vector<float> inputs(3);
// 	
// 	const float asdas  = 0.3;
// 	inputs[0] = asdas;
// 	const float input2 = 2.5;
// 	inputs[1] = input2;
// 	const float input3 = 1.8;
// 	inputs[2] = input3;
// 	. . .
// 

// Inputs Names:
// Artificial Intelligence Techniques SL	
// artelnics@artelnics.com	
// Your model has been exported to this c file.
// You can manage it with the main method, where you 	
// can change the values of your inputs. For example:
// if we want to add these 3 values (0.3, 2.5 and 1.8)
// to our 3 inputs (Input_1, Input_2 and Input_1), the
// main program has to look like this:
// 	
// int main(){ 
// 	vector<float> inputs(3);
// 	
// 	const float asdas  = 0.3;
// 	inputs[0] = asdas;
// 	const float input2 = 2.5;
// 	inputs[1] = input2;
// 	const float input3 = 1.8;
// 	inputs[2] = input3;
// 	. . .
// 

// Inputs Names:
	0) frequency
	1) angle_of_attack
	2) cho_rd_lenght
	3) free_res_stream_velocity
	4) suction_side_di_splacement_thickness


#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>


using namespace std;


vector<float> calculate_outputs(const vector<float>& inputs)
{
	const float frequency = inputs[0];
	const float angle_of_attack = inputs[1];
	const float cho_rd_lenght = inputs[2];
	const float free_res_stream_velocity = inputs[3];
	const float suction_side_di_splacement_thickness = inputs[4];

	double scaled_frequency = (frequency-2886.380615)/3152.573242;
	double scaled_angle_of_attack = (angle_of_attack-6.782301903)/5.918128014;
	double scaled_cho_rd_lenght = (cho_rd_lenght-0.136548236)/0.09354072809;
	double scaled_free_res_stream_velocity = (free_res_stream_velocity-50.86074448)/15.57278538;
	double scaled_suction_side_di_splacement_thickness = (suction_side_di_splacement_thickness-0.01113987993)/0.01315023471;

	double perceptron_layer_output_0 = ( 0.0160486 + (scaled_frequency*-0.581589) + (scaled_angle_of_attack*-0.368824) + (scaled_cho_rd_lenght*-0.497451) + (scaled_free_res_stream_velocity*0.218891) + (scaled_suction_side_di_splacement_thickness*-0.267632));

	double unscaling_layer_output_0=perceptron_layer_output_0*6.898656845+124.8359451;

	double scaled_sound_pressure_level = max(-3.402823466e+38, unscaling_layer_output_0);
	scaled_sound_pressure_level = min(3.402823466e+38, scaled_sound_pressure_level);

	vector<float> out(1);
	out[0] = scaled_sound_pressure_level;

	return out;
}


int main(){ 

	vector<float> inputs(5); 

	const float frequency = /*enter your value here*/; 
	inputs[0] = frequency;
	const float angle_of_attack = /*enter your value here*/; 
	inputs[1] = angle_of_attack;
	const float cho_rd_lenght = /*enter your value here*/; 
	inputs[2] = cho_rd_lenght;
	const float free_res_stream_velocity = /*enter your value here*/; 
	inputs[3] = free_res_stream_velocity;
	const float suction_side_di_splacement_thickness = /*enter your value here*/; 
	inputs[4] = suction_side_di_splacement_thickness;

	vector<float> outputs(1);

	outputs = calculate_outputs(inputs);

	printf("These are your outputs:\n");
	printf( "scaled_sound_pressure_level: %f \n", outputs[0]);

	return 0;
} 

