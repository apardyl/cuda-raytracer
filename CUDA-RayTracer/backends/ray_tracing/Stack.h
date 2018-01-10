#pragma once

struct Stack {
	int tab[60];
	int size = 0;

	Stack();

	void add_element(int x);

	int top();

	void pop();
};