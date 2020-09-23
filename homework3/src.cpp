/*
-실험 과제
(1) AND, OR, XOR 구분 실험
(2) 도우넛 모양 구분 실험(아래 데이터 이용)

- Layer 수, Layer 당 node 수는 변수로 지정할 것. ㅇㅋ
- weight는 행렬 형식으로 파일에 저장 ㅇㅋ
- Learning 과정을 그래프로 보여주기(X1, X2 2차원 직선 그래프).생략
대신 귀여운 결과 그래프를 드리겠습니다
- 각 노드마다 직선을 그림으로 표시. 생략
대신 귀여운 고양이를 드리겠습니다
- iteration에 따른 Error 그래프 ㅇㅋ
- 구현언어: C, C++ ㅇㅋ
- 제출물 : 프로그램, 결과 보고서 ㅇㅋ
실행 10 %, 출력 10, 주석 10, 완성도 25, 오류 10, 창의 10 보고서 25 % ㅇㅋ

-도우넛 모양 데이터 ㅇㅋ

*/

/*

	2019-2 인공지능 과제3 코드
	2015920003 컴퓨터과학부 김건호

*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "NeuralNetwork.h"

int main(void)
{
	// datasets for training
	float *training_input_GATE[4] = {
		new float[2]{ 0, 0 },
		new float[2]{ 0, 1 },
		new float[2]{ 1, 0 },
		new float[2]{ 1, 1 }
	};
	float *training_input_DN[9] = {
		new float[2]{   0,   0 },
		new float[2]{   0,   1 },
		new float[2]{   1,   0 },
		new float[2]{   1,   1 },
		new float[2]{ 0.5,   1 },
		new float[2]{   1, 0.5 },
		new float[2]{   0, 0.5 },
		new float[2]{ 0.5,   0 },
		new float[2]{ 0.5, 0.5 }
	};
	float *training_output_AND[4] = {
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 1 }
	};
	float *training_output_OR[4] = {
		new float[1]{ 0 },
		new float[1]{ 1 },
		new float[1]{ 1 },
		new float[1]{ 1 }
	};
	float *training_output_XOR[4] = {
		new float[1]{ 0 },
		new float[1]{ 1 },
		new float[1]{ 1 },
		new float[1]{ 0 }
	};
	float *training_output_DN[9] = {
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 0 },
		new float[1]{ 1 }
	};

	// to set perceptrons's weights and threshold randomly, perform srand() on main()
	srand(time(NULL));
	rand();

	// create and initialize neural networks
	NeuralNetwork neural_network_AND(4, 2, 3, 3, 1);
	NeuralNetwork neural_network_OR(3, 2, 3, 1);
	NeuralNetwork neural_network_XOR(3, 2, 4, 1);
	NeuralNetwork neural_network_DN(5, 2, 3, 3, 3, 1);

	// train the networks and log the results
	neural_network_AND.Train(
		4,                   // length of training dataset
		training_input_GATE, // training dataset
		training_output_AND, // training dataset
		0.02f,               // tolerance
		0.75f,               // momentum
		"threshold_AND.txt", // file name to log the thresholds
		"weight_AND.txt",    // file name to log the weights
		"error_AND.txt"      // file name to log the error
	);
	neural_network_OR.Train(
		4,                   // length of training dataset
		training_input_GATE, // training dataset
		training_output_OR,  // training dataset
		0.02f,               // tolerance
		0.75f,               // momentum
		"threshold_OR.txt",  // file name to log the thresholds
		"weight_OR.txt",     // file name to log the weights
		"error_OR.txt"       // file name to log the error
	);
	neural_network_XOR.Train(
		4,                   // length of training dataset
		training_input_GATE, // training dataset
		training_output_XOR, // training dataset
		0.02f,               // tolerance
		0.75f,               // momentum
		"threshold_XOR.txt", // file name to log the thresholds
		"weight_XOR.txt",    // file name to log the weights
		"error_XOR.txt"      // file name to log the error
	);
	neural_network_DN.Train(
		9,                   // length of training dataset
		training_input_DN,   // training dataset
		training_output_DN,  // training dataset
		0.02f,               // tolerance
		0.75f,               // momentum
		"threshold_DN.txt",  // file name to log the thresholds
		"weight_DN.txt",     // file name to log the weights
		"error_DN.txt"       // file name to log the error
	);

	// wait for end
	printf("Enter to exit\n");
	getchar();
}