#include "Stack.h"

Stack::Stack()
{
	
}


void Stack::add_element(int x)
{
	tab[size++] = x;
}

int Stack::top()
{
	return tab[size - 1];
}

void Stack::pop()
{
	size--;
}
