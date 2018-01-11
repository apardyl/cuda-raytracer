#ifndef RAY_TRACER_STACK_H
#define RAY_TRACER_STACK_H

struct Stack {
    int tab[60];
    int size = 0;

    Stack();

    void addElement(int x);

    int top();

    void pop();
};
#endif //RAY_TRACER_STACK_H
