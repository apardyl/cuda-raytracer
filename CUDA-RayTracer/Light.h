#include<Models.h>


struct Light {
	Point point;
	Color Is, Id;

	Light();

	Light(Point point, Color Is, Color Id);
};

