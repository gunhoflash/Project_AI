/*

	2019-2 인공지능 과제1 코드
	2015920003 컴퓨터과학부 김건호

*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N 2                              // the input's demension of perceptron
#define LOWERBOUND_WEIGHT -1             // lowerbound of weights
#define UPPERBOUND_WEIGHT  1             // upperbound of weights
#define LOWERBOUND_THRESHOLD 0           // lowerbound of threshold
#define UPPERBOUND_THRESHOLD N / 2.0     // upperbound of threshold

float weight[N];
float threshold;

// return a random float number between lowerbound and upperbound
float rand_float(float lowerbound, float upperbound)
{
	return lowerbound + ((rand() % 1000 + 1) / 1000.0) * (upperbound - lowerbound);
}

// Set the weights with random numbers
void randomize_weights()
{
	srand(time(NULL));
	for (int i = 0; i < N; i++)
		weight[i] = rand_float(LOWERBOUND_WEIGHT, UPPERBOUND_WEIGHT);
}

// Set the threshold with random number
void randomize_threshold()
{
	srand(time(NULL));
	threshold = rand_float(LOWERBOUND_THRESHOLD, UPPERBOUND_THRESHOLD);
}

// Set the weights with user input
void input_weights()
{
	for (int i = 0; i < N; i++)
	{
		printf("Input weight[%d]: ", i);
		scanf("%f", &weight[i]);
		if (weight[i] < LOWERBOUND_WEIGHT) weight[i] = LOWERBOUND_WEIGHT;
		if (weight[i] > UPPERBOUND_WEIGHT) weight[i] = UPPERBOUND_WEIGHT;
	}
}

// Set the threshold with user input
void input_threshold()
{
	printf("Input threshold: ");
	scanf("%f", &threshold);
	if (threshold < LOWERBOUND_THRESHOLD) threshold = LOWERBOUND_THRESHOLD;
	if (threshold > UPPERBOUND_THRESHOLD) threshold = UPPERBOUND_THRESHOLD;
}

// 1-layer perceptron
int perceptron_1_layer(int input[N])
{
	float net = -threshold;
	for (int i = 0; i < N; i++)
		net += input[i] * weight[i];
	return (net > 0) ? 1 : 0;
}

int main(void)
{
	// init
	int wrong;

	// Randomize the weights and threshold
	randomize_weights();
	randomize_threshold();

	// Find appropriate weights and threshold
	while (1)
	{
		wrong = 0;

		if (perceptron_1_layer((int[]) { 0, 0 }) != 0) wrong++; // perceptron 0 0
		if (perceptron_1_layer((int[]) { 0, 1 }) != 0) wrong++; // perceptron 0 1
		if (perceptron_1_layer((int[]) { 1, 0 }) != 0) wrong++; // perceptron 1 0
		if (perceptron_1_layer((int[]) { 1, 1 }) != 1) wrong++; // perceptron 1 1

		// Break this loop when nothing is wrong
		if (wrong == 0)
		{
			printf("Correct\n");
			break;
		}
		else printf("Wrong: %d\n", wrong);

		/*
			Change the weights with user inputs.
			Don't change the threshold.
		*/
		input_weights();
		// randomize_weights();
		// input_threshold();
		// randomize_threshold();
	}
}