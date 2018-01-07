#include "Material.h"


Material::Material() {}

Material::Material(float Ks, float Kd, float Ka, float alfa) {
	this->Ks = Ks;
	this->Kd = Kd;
	this->Ka = Ka;
	this->alfa = alfa;
}

