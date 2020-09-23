#include <errno.h> 
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#define MAX_LR 0.2f
#define MIN_LR 0.02f
#define MAX_EPOCH 1048576 // 262144

class NeuralNetwork
{
private:

	/*
		properties
	*/

	int      number_of_inputs;      // number of inputs
	int      number_of_outputs;     // number of outputs
	int      number_of_layers;      // number of layers (exclude input layer)
	int     *sizes_of_layer;        // number of perceptrons of each layer (include input layer)
	int     *number_of_perceptrons; // number of perceptrons of each layer

	float    learning_rate;         // dynamic learning rate
	float    weight_momentum;       // momentum
	
	float  **thresholds;            // thresholds
	float ***weights;               // weights
	float ***momentum;              // calculated momentums for each weights
	float  **results;               // calculated result from each perceptrons
	float  **delta;                 // calculated delta to adjust each weights
	float   *revised_output;        // final results of network

	/*
		private functions
	*/

	// private functions for malloc
	float   *MallocFloat(int length);
	float  **MallocFloatPointer(int length);
	float ***MallocFloatPointerPointer(int length);
	int     *MallocInt(int length);

	// private functions for the activation function
	float    Activation(float f);
	float    DActivation(float f);
	
	// private functions for loss calculation and feedback
	float    Loss(float *input, float *output, float *expect);
	void     Feedback(float *input, float *output, float *expect);

	// private functions for output revision
	void     ReviseOutput(int type_of_revision);
	void     SoftMax();
	void     OneHot();

	// private function for initialization
	float    RandomWeight(int input, int output);

public:

	/*
		public functions
	*/

	// constructor
	NeuralNetwork(int args, ...);

	// public function for initialization
	void     Init();

	// public functions for calculation and training
	float   *Calculate(float *input, int type_of_revision);
	void     Train(int length, float **inputs, float **outputs, float tolerance, float momentum, const char *filename_threshold, const char *filename_weight, const char *filename_error);

	// public functions for print
	void     PrintThresholds();
	void     PrintWeights();
	void     FPrintThresholds(FILE *file);
	void     FPrintWeights(FILE *file);
};

/*
	Description:
		Perform malloc() for float array.

	Parameters:
		(none)

	Return:
		(memory address)
*/
float *NeuralNetwork::MallocFloat(int length)
{
	float *p = NULL;
	while (p == NULL)
		p = (float *)malloc(sizeof(float) * length);
	return p;
}

/*
	Description:
		Perform malloc() for float * array.

	Parameters:
		(none)

	Return:
		(memory address)
*/
float **NeuralNetwork::MallocFloatPointer(int length)
{
	float **p = NULL;
	while (p == NULL)
		p = (float **)malloc(sizeof(float *) * length);
	return p;
}

/*
	Description:
		Perform malloc() for float ** array.

	Parameters:
		(none)

	Return:
		(memory address)
*/
float ***NeuralNetwork::MallocFloatPointerPointer(int length)
{
	float ***p = NULL;
	while (p == NULL)
		p = (float ***)malloc(sizeof(float **) * length);
	return p;
}

/*
	Description:
		Perform malloc() for int array.

	Parameters:
		(none)

	Return:
		(memory address)
*/
int *NeuralNetwork::MallocInt(int length)
{
	int *p = NULL;
	while (p == NULL)
		p = (int *)malloc(sizeof(int) * length);
	return p;
}

/*
	Description:
		Activation function

	Parameters:
		f: input of function

	Return:
		(calculated result)
*/
float NeuralNetwork::Activation(float f)
{
	// return 1 / (1 + (float)exp(-f)); // sigmoid
	return (f > 0) ? f : (float) exp(f) - 1; // ELU
}

/*
	Description:
		Differentiate value of activation function

	Parameters:
		f: input of function (result of activation function)

	Return:
		(calculated result)
*/
float NeuralNetwork::DActivation(float f)
{
	// return f * (1 - f); // sigmoid
	return (f > 0) ? 1 : f + 1; // ELU
}

/*
	Description:
		Calculate cost: Quadratic Loss Function

	Parameters:
		input: input of training-data
		output: calculated output
		expect: expected output

	Return:
		(cost)
*/
float NeuralNetwork::Loss(float *input, float *output, float *expect)
{
	int i;
	float error;

	for (i = error = 0; i < number_of_outputs; i++)
		error += pow(output[i] - expect[i], 2);

	return error;
}

/*
	Description:
		Adjust the weight values with training result.

	Parameters:
		input: input of training-data
		output: calculated output
		expect: expected output

	Return:
		(none)
*/
void NeuralNetwork::Feedback(float *input, float *output, float *expect)
{
	int i, j, k;
	float dw, delta_weight;

	// calculate deltas of the last layer
	for (i = 0; i < sizes_of_layer[number_of_layers]; i++)
	{
		delta[number_of_layers - 1][i] = (output[i] - expect[i]) * DActivation(results[number_of_layers][i]);
	}

	// calculate deltas of the previous layers
	for (i = number_of_layers - 2; i >= 0; i--)
	{
		for (j = 0; j < sizes_of_layer[i + 1]; j++)
		{
			dw = 0;
			for (k = 0; k < sizes_of_layer[i + 2]; k++)
			{
				dw += delta[i + 1][k] * weights[i + 1][k][j];
			}
			delta[i][j] = dw * DActivation(results[i + 1][j]);
		}
	}

	// adjust momentums and weights
	for (i = 0; i < number_of_layers; i++)
	{
		for (j = 0; j < sizes_of_layer[i]; j++)
		{
			for (k = 0; k < sizes_of_layer[i + 1]; k++)
			{
				delta_weight = -learning_rate * delta[i][k] * results[i][j];
				momentum[i][k][j] *= weight_momentum;
				momentum[i][k][j] += delta_weight;
				weights[i][k][j] += momentum[i][k][j];
			}
		}
	}
}

/*
	Description:
		Revise the calculation result.

	Parameters:
		type_of_revision: define the type of revision

	Return:
		(none)
*/
void NeuralNetwork::ReviseOutput(int type_of_revision)
{
	switch (type_of_revision)
	{
		case 0:            break; // none
		case 1: SoftMax(); break; // softmax
		case 2: OneHot();  break; // one hot
		default:           break; // unknown
	}
}

/*
	Description:
		Revise the calculation result: softmax

	Parameters:
		(none)

	Return:
		(none)
*/
void NeuralNetwork::SoftMax()
{
	int i;
	float sum;
	
	// set all negative value as 0
	for (i = sum = 0; i < number_of_outputs; i++)
	{
		if (results[number_of_layers][i] < 0)
			results[number_of_layers][i] = 0;
		sum += results[number_of_layers][i];
	}

	// handle exception
	if (sum == 0) return;

	// make total sum as 1
	for (i = 0; i < number_of_outputs; i++)
		results[number_of_layers][i] /= sum;
}

/*
	Description:
		Revise the calculation result: one hot

	Parameters:
		(none)

	Return:
		(none)
*/
void NeuralNetwork::OneHot()
{
	int i, max = -1;
	float sum;

	// set all negative value as 0
	for (i = sum = 0; i < number_of_outputs; i++)
	{
		if (results[number_of_layers][i] <= 0)
			results[number_of_layers][i] = 0;
		else max = i;
	}

	// handle exception
	if (max == -1) return;

	// find the index with maximum value
	for (i = 1; i < number_of_outputs; i++)
		if (results[number_of_layers][i] > results[number_of_layers][max])
			max = i;

	// set all to 0 except output[max]
	for (i = 0; i < number_of_outputs; i++)
		results[number_of_layers][i] = 0;
	results[number_of_layers][max] = 1;
}

/*
	Description:
		Return a random float.
		Median numbers are more likely to be returned than boundary numbers.

	Parameters:
		input: the size of previous layer
		output: the size of next layer

	Return:
		(random float)
*/
float NeuralNetwork::RandomWeight(int input, int output)
{
	float r = ((float)rand()) / RAND_MAX * 2 - 1; // r is a random float from -1 to 1
	return r * sqrt(1.0f / (input + output));
}

/*
	Description:
		Constructor.
		Allocate memory and initialize variables.

	Parameters:
		args: number of layer (include input/output layer)
		...

	Return:
		(none)
*/
NeuralNetwork::NeuralNetwork(int args, ...)
{
	// handle exception
	if (args < 2)
	{
		perror("Invalid parameters");
		return;
	}

	int i, j;
	va_list ap;
	va_start(ap, args);

	// set variables
	weight_momentum  = 0;
	number_of_layers = args - 1;

	// malloc for neural network
	sizes_of_layer   = MallocInt(number_of_layers + 1); // include inputs
	weights          = MallocFloatPointerPointer(number_of_layers);
	momentum         = MallocFloatPointerPointer(number_of_layers);
	thresholds       = MallocFloatPointer(number_of_layers);
	results          = MallocFloatPointer(number_of_layers + 1); // include inputs
	delta            = MallocFloatPointer(number_of_layers);

	// get dimensions of each layer (include the number of inputs)
	for (i = 0; i < number_of_layers + 1; i++)
		sizes_of_layer[i] = va_arg(ap, int);
	va_end(ap);
	number_of_inputs  = sizes_of_layer[0];
	number_of_outputs = sizes_of_layer[i - 1];

	// malloc for each layer
	for (i = 0; i < number_of_layers; i++)
	{
		weights[i]         = MallocFloatPointer(sizes_of_layer[i + 1]);
		momentum[i]        = MallocFloatPointer(sizes_of_layer[i + 1]);
		for (j = 0; j < sizes_of_layer[i + 1]; j++)
		{
			weights[i][j]  = MallocFloat(sizes_of_layer[i]);
			momentum[i][j] = MallocFloat(sizes_of_layer[i]);
		}
		thresholds[i]      = MallocFloat(sizes_of_layer[i + 1]);
		results[i]         = MallocFloat(sizes_of_layer[i]);
		delta[i]           = MallocFloat(sizes_of_layer[i + 1]);
	}
	results[i]             = MallocFloat(sizes_of_layer[i]);

	// initialize thresholds and weights
	Init();
}

/*
	Description:
		Initialize variables.

	Parameters:
		(none)

	Return:
		(none)
*/
void NeuralNetwork::Init()
{
	int i, j, k;

	// initialize each values(thresholds, weights, momentums, results, deltas)
	for (i = 0; i < number_of_layers; i++)
		for (j = 0; j < sizes_of_layer[i + 1]; j++)
		{
			for (k = 0; k < sizes_of_layer[i]; k++)
			{
				weights[i][j][k]  = RandomWeight(sizes_of_layer[i], sizes_of_layer[i + 1]);
				momentum[i][j][k] = 0;
			}
			thresholds[i][j]  = 0;
			results[i + 1][j] = 0;
			delta[i][j]       = 0;
		}

	printf("Neural network initialized.\n");
}

/*
	Description:
		Calculate the output for the given inputs.

	Parameters:
		input: [number_of_inputs]
		type_of_revision: define the type of revision

	Return:
		(results)
*/
float *NeuralNetwork::Calculate(float *input, int type_of_revision)
{
	int i, j, k;

	// copy inputs
	for (i = 0; i < number_of_inputs; i++)
		results[0][i] = input[i];

	// calculate each layer, each perceptron
	for (i = 1; i < number_of_layers + 1; i++)
		for (j = 0; j < sizes_of_layer[i]; j++)
		{
			// calculate each net
			results[i][j] = -thresholds[i - 1][j];
			for (k = 0; k < sizes_of_layer[i - 1]; k++)
				results[i][j] += results[i - 1][k] * weights[i - 1][j][k];
			results[i][j] = Activation(results[i][j]);
		}

	// revise
	ReviseOutput(type_of_revision);

	return results[number_of_layers];
}

/*
	Description:
		Train neural network with training dataset.
		Every repeated round:
			1. Calculate output and loss.
			2. If the loss is small, end training.
			3. If the loss is big, feedback(adjust the weights) and log the error.
			4. Repeat above process.

	Parameters:
		length: length of training dataset
		inputs: [length][number_of_inputs]
		outputs: [length][number_of_outputs]
		tolerance: loss tolerance
		filename_threshold: file name to log the thresholds
		filename_weight: file name to log the weights
		filename_error: file name to log the error

	Return:
		(none)
*/
void NeuralNetwork::Train(int length, float **inputs, float **outputs, float tolerance, float momentum, const char *filename_threshold, const char *filename_weight, const char *filename_error)
{
	FILE *file_threshold = NULL;
	FILE *file_weight    = NULL;
	FILE *file_error     = NULL;
	int i, epoch;
	bool trainable;
	float *input;
	float *output;
	float *expect;
	float loss, loss_max, loss_sum;

	// handle exception
	if (length < 1)
	{
		printf("No data to train.\n");
		return;
	}
	if (tolerance < 0) tolerance = 0;

	// open the files to write
	if (filename_threshold) file_threshold = fopen(filename_threshold, "w");
	if (filename_weight)    file_weight    = fopen(filename_weight, "w");
	if (filename_error)     file_error     = fopen(filename_error,  "w");
	
	// can't open the files
	if (filename_threshold && file_threshold == NULL) printf("Can't open the file: %s.\n", filename_threshold);
	if (filename_weight    && file_weight    == NULL) printf("Can't open the file: %s.\n", filename_weight);
	if (filename_error     && file_error     == NULL) printf("Can't open the file: %s.\n", filename_error);

	// no file
	if (file_threshold == NULL) printf("Train without log the threshold.\n");
	if (file_weight    == NULL) printf("Train without log the weight.\n");
	if (file_error     == NULL) printf("Train without log the error.\n");

	// init
	epoch     = 0;
	trainable = true;
	this->weight_momentum = momentum;
	printf("Start training\n");

	// train repeatedly
	while (trainable)
	{
		trainable = false;

		// if tried enough, end training
		if (++epoch == MAX_EPOCH)
		{
			printf("Tried enough but didn't be trained.\n");
			break;
		}
		learning_rate = MAX_LR - (MAX_LR - MIN_LR) * epoch / MAX_EPOCH;

		// for each training data,
		for (i = loss_max = loss_sum = 0; i < length; i++)
		{
			// calculate output and loss
			input  = inputs[i];
			output = Calculate(input, 0);
			expect = outputs[i];
			loss   = Loss(input, output, expect);

			if (loss_max < loss) loss_max = loss;
			loss_sum += loss;

			// compare result with expected output
			if (loss > tolerance)
			{
				// train one more time
				trainable = true;

				// adjust the weight values
				Feedback(input, output, expect);
			}
		}

		// log the maximum error
		if (file_error)
			fprintf(file_error, "%.6f\t%.6f\n", loss_max, loss_sum / length);

		if (!trainable)
			printf("Trained with %d epoch.\n", epoch);
	}

	// log the thresholds and weights
	FPrintThresholds(file_threshold);
	FPrintWeights(file_weight);

	// close files
	if (file_threshold) fclose(file_threshold);
	if (file_weight)    fclose(file_weight);
	if (file_error)     fclose(file_error);
}

/*
	Description:
		Print thresholds to standard output.

	Parameters:
		(none)

	Return:
		(none)
*/
void NeuralNetwork::PrintThresholds()
{
	int i, j;
	for (i = 0; i < number_of_layers; i++)
	{
		for (j = 0; j < sizes_of_layer[i + 1]; j++)
			printf("[%d][%d]:%+.16f ", i, j, thresholds[i][j]);
		printf("\n");
	}
}

/*
	Description:
		Print weights to standard output.

	Parameters:
		(none)

	Return:
		(none)
*/
void NeuralNetwork::PrintWeights()
{
	int i, j, k;
	for (i = 0; i < number_of_layers; i++)
	{
		for (j = 0; j < sizes_of_layer[i + 1]; j++)
		{
			for (k = 0; k < sizes_of_layer[i]; k++)
				printf("[%d][%d][%d]:%+.16f ", i, j, k, weights[i][j][k]);
			printf("\n");
		}
		printf("\n");
	}
}

/*
	Description:
		Print thresholds to file.

	Parameters:
		(none)

	Return:
		(none)
*/
void NeuralNetwork::FPrintThresholds(FILE *file)
{
	// handle exception
	if (!file) return;

	int i, j;
	for (i = 0; i < number_of_layers; i++)
	{
		for (j = 0; j < sizes_of_layer[i + 1]; j++)
			fprintf(file, "[%d][%d]:%+.16f ", i, j, thresholds[i][j]);
		fprintf(file, "\n");
	}
}

/*
	Description:
		Print thresholds to weights to file.

	Parameters:
		(none)

	Return:
		(none)
*/
void NeuralNetwork::FPrintWeights(FILE *file)
{
	// handle exception
	if (!file) return;

	int i, j, k;
	for (i = 0; i < number_of_layers; i++)
	{
		for (j = 0; j < sizes_of_layer[i + 1]; j++)
		{
			for (k = 0; k < sizes_of_layer[i]; k++)
				fprintf(file, "[%d][%d][%d]:%+.16f ", i, j, k, weights[i][j][k]);
			fprintf(file, "\n");
		}
		fprintf(file, "\n");
	}
}