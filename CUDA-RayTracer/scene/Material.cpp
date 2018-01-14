#include "Material.h"

Material::Material(): specularExponent(1), dissolve(1), refractiveIndex(1) {
}

Material::Material(Color ambient, Color diffuse, Color specular, float specularExponent,
                   float dissolve)
    : ambient(ambient), diffuse(diffuse), specular(specular), specularExponent(specularExponent),
      dissolve(dissolve), refractiveIndex(1) {
}

Material::Material(Color ambient, Color diffuse, Color specular, Color transparent,
                   float specularExponent, float dissolve, float refractiveIndex) :
    ambient(ambient), diffuse(diffuse), specular(specular), transparent(transparent),
    specularExponent(specularExponent),
    dissolve(dissolve), refractiveIndex(refractiveIndex) {
}
