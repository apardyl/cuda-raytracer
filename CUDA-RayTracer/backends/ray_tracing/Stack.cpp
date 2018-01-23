#include "Stack.h"

Stack::Stack() = default;

void Stack::addElement(int x) {
    tab[size++] = x;
}

int Stack::top() {
    return tab[size - 1];
}

void Stack::pop() {
    size--;
}

bool Stack::isEmpty() {
    return size == 0;
}
