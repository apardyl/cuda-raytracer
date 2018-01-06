#include "Material.h"

Material::Material(Color ambient, Color diffuse, Color specular, float specularExponent, float dissolve)
    : ambient(ambient), diffuse(diffuse), specular(specular), specularExponent(specularExponent), dissolve(dissolve) {
}
