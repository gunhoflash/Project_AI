#define INITIAL_LEARNING_RATE 0.26f
class Perceptron
{
private:
	/* Member Variables */
	int n;                           // dimension of perceptron
	float *weights;                  // weight values
	float *before_weights;           // saved weight values before training
	float threshold;                 // threshold
	float C = INITIAL_LEARNING_RATE; // learning rate

	/* Private Methods */
	float *MallocForWeights();
	void SaveWeights();
	int CalculateTrainingDataset(int length, int **inputs, int *outputs, bool feedback);
	void Feedback(int *input, int output, int expect);

public:
	/* Public Methods */
	Perceptron(int n);
	int Calculate(int *input);
	int Train(int length, int **input, int *output);
	void PrintWeights();
};

/*
	Description:
		Perform malloc() for weight values.

	Parameters:
		(none)

	Return:
		(memory address)
*/
float *Perceptron::MallocForWeights()
{
	float *p = NULL;
	while (p == NULL)
		p = (float *)malloc(sizeof(float) * n);
	return p;
}

/*
	Description:
		Save current weight values before training.

	Parameters:
		(none)

	Return:
		(none)
*/
void Perceptron::SaveWeights()
{
	for (int i = 0; i < n; i++)
		before_weights[i] = weights[i];
}

/*
	Description:
		Calculate and count the number of errors.
		If necessary, readjust the weight values by feedback.

	Parameters:
		length: length of training-dataset
		inputs: [length][n]
		outputs: [length]
		feedback: if true, readjust the weight values

	Return:
		(the number of errors)
*/
int Perceptron::CalculateTrainingDataset(int length, int **inputs, int *outputs, bool feedback)
{
	int i, error;
	int *input;
	int output;
	int expect;

	// for each training data,
	for (i = error = 0; i < length; i++)
	{
		// calculate
		input  = inputs[i];
		output = Calculate(input);
		expect = outputs[i];

		// if wrong,
		if (output != expect)
		{
			// count error up
			error++;
			// and readjust the weight values
			if (feedback)
				Feedback(input, output, expect);
		}
	}

	// return the number of errors
	return error;
}

/*
	Description:
		Readjust the weight values with training result.

	Parameters:
		input: input of training-data
		output: calculated output
		expect: expected output

	Return:
		(none)
*/
void Perceptron::Feedback(int *input, int output, int expect)
{
	for (int i = 0; i < n; i++)
		weights[i] += C * (expect - output) * 1 * input[i];
}

/*
	Description:
		Constructor.
		Initialize the weights and threshold randomly.
			weights: from -1 to 1
			threshold: from 0 to 1 (not 0)

	Parameters:
		N: dimension of perceptron

	Return:
		(none)
*/
Perceptron::Perceptron(int N)
{
	n = N;
	weights = MallocForWeights();
	before_weights = MallocForWeights();

	// initialize the weights and threshold
	for (int i = 0; i < n; i++)
		weights[i] = (rand() % 1001) / 500.0f - 1;
	threshold = (rand() % 1000 + 1) / 1000.0f;
}

/*
	Description:
		Calculate the output for the given inputs.

	Parameters:
		input: [n]

	Return:
		0
		1
*/
int Perceptron::Calculate(int *input)
{
	float net = -threshold;
	for (int i = 0; i < n; i++)
		net += input[i] * weights[i];
	return (net > 0) ? 1 : 0;
}

/*
	Description:
		Train perceptron with training dataset.
		Every repeated round:
			1. Save current weight values.
			2. Calculate and feedback(readjust the weights).
			3. Count the number of errors.
			4. If error is 0, end.
			5. If not, repeat above process (and reduce the learning rate when needed).

	Parameters:
		length: length of training-dataset
		inputs: [length][n]
		outputs: [length]

	Return:
		0: trained
		1: not trained
*/
int Perceptron::Train(int length, int **inputs, int *outputs)
{
	bool trainable;
	int i, error, round;

	// handle exception
	if (length < 1)
	{
		printf("\nNo data to train.\n\n");
		return 1;
	}

	// print threshold, weights, and errors before training
	printf("threshold: %8f\n", threshold);
	PrintWeights();
	error = CalculateTrainingDataset(length, inputs, outputs, false);
	printf("\nError: %d\n\n", error);

	// train
	round = 1;
	trainable = true;
	while (trainable)
	{
		printf("[Round %03d]\t", round++);

		// save current weight values
		SaveWeights();

		// calculate and feedback
		CalculateTrainingDataset(length, inputs, outputs, true);

		// print readjusted weight values
		PrintWeights();

		// calculate and count the number of errors
		error = CalculateTrainingDataset(length, inputs, outputs, false);

		// print the number of errors
		printf("\tError: %d\n", error);

		// end training when all is correct :)
		if (error == 0)
		{
			printf("\nTrained successfully.\n\n");
			return 0;
		}
		else
		{
			// check if the weight value has changed
			trainable = false;
			for (i = 0; i < n; i++)
				if (weights[i] != before_weights[i])
					trainable = true;

			/*
				Note:
					if you want to apply 'Constant Learning Rate',
					just perform 'continue' right here.
			*/
			// continue;

			// or apply 'Dynamic Learning Rate'
			if (!trainable)
			{
				// if no weigth was changed,
				// down the learning rate
				C /= 2;
				/*
					If the value of C is small enough for
					the computer to recognize it as zero,
					training fail...
				*/
				if (C == 0) C = INITIAL_LEARNING_RATE; // training may end... :(
				else
				{
					printf("\n[Reduce Learning Rate: %g]\n\n", C);
					trainable = true;
				}
			}
		}
	}

	printf("\nNot trained.\n\n");
	return 1;
}

/*
	Description:
		Print all weight values

	Parameters:
		(none)

	Return:
		(none)
*/
void Perceptron::PrintWeights()
{
	printf("weight:");
	for (int i = 0; i < n; i++)
		printf(" %+8f", weights[i]);
}