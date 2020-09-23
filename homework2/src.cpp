/*

	2019-2 인공지능 과제2 코드
	2015920003 컴퓨터과학부 김건호

*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "Perceptron.h"

int main(void)
{
	// Dataset for training
	int *training_input[4] = {
		new int[2]{ 0, 0 },
		new int[2]{ 0, 1 },
		new int[2]{ 1, 0 },
		new int[2]{ 1, 1 }
	};
	int training_output_AND[4] = { 0, 0, 0, 1 };
	int training_output_OR[4]  = { 0, 1, 1, 1 };
	int training_output_XOR[4] = { 1, 0, 0, 1 };

	// To set perceptrons's weights and threshold randomly, perform srand() on main()
	srand(time(NULL));
	rand();

	// Initialize perceptrons
	Perceptron perceptron_AND(2);
	Perceptron perceptron_OR(2);
	Perceptron perceptron_XOR(2);

	// Train AND-gate
	printf("Train AND\n");
	perceptron_AND.Train(4, training_input, training_output_AND);

	// Train OR-gate
	printf("Train OR\n");
	perceptron_OR.Train(4, training_input, training_output_OR);

	// Train XOR-gate
	printf("Train XOR\n");
	perceptron_XOR.Train(4, training_input, training_output_XOR);

	// Wait for end
	printf("Enter to exit\n");
	getchar();
}