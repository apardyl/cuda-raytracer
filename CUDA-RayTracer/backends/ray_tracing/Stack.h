#ifndef RAY_TRACER_STACK_H
#define RAY_TRACER_STACK_H

#include <cstddef>

class Stack {
private:
    int tab[60];
    size_t size = 0;

public:
    Stack();

    void addElement(int x);

    bool isEmpty();

    int top();

    void pop();
};

#endif //RAY_TRACER_STACK_H
