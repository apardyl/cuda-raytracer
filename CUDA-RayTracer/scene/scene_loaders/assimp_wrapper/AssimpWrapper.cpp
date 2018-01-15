#include "AssimpWrapper.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "scene/Scene.h"
#include "scene/scene_loaders/ParseError.h"

Color AssimpWrapper::getColor(const aiMaterial *material, const char *pKey, unsigned type,
                              unsigned idx, Color defaulColor) {
    aiColor3D color3(0.f, 0.f, 0.f);
    if (material->Get(pKey, type, idx, color3)) {
        return defaulColor;
    }
    return {color3.r, color3.g, color3.b};
}

float AssimpWrapper::getFloat(const aiMaterial *material, const char *pKey, unsigned type,
                              unsigned idx, float color) {
    material->Get(pKey, type, idx, color);
    return color;
}

Point AssimpWrapper::getPoint(const aiMesh *mesh, unsigned idx) {
    return {mesh->mVertices[idx].x, mesh->mVertices[idx].y, mesh->mVertices[idx].z};
}

Vector AssimpWrapper::getNormal(const aiMesh *mesh, unsigned idx) {
    return Vector(Point(0.f, 0.f, 0.f),
                  mesh->mNormals[idx].x, mesh->mNormals[idx].y, mesh->mNormals[idx].z)
            .normalize();
}

void AssimpWrapper::loadMaterials() {
    materials = new Material[loadedScene->mNumMaterials];
    materialsCount = loadedScene->mNumMaterials;
    for (int i = 0; i < loadedScene->mNumMaterials; i++) {
        const aiMaterial *material = loadedScene->mMaterials[i];
        materials[i].ambient = getColor(material, AI_MATKEY_COLOR_AMBIENT);
        materials[i].diffuse = getColor(material, AI_MATKEY_COLOR_DIFFUSE);
        materials[i].specular = getColor(material, AI_MATKEY_COLOR_SPECULAR);
        materials[i].transparent = getColor(material, AI_MATKEY_COLOR_TRANSPARENT, Color(1, 1, 1));
        materials[i].specularExponent = getFloat(material, AI_MATKEY_SHININESS) / 4;
        materials[i].dissolve = getFloat(material, AI_MATKEY_OPACITY);
        materials[i].refractiveIndex = getFloat(material, AI_MATKEY_REFRACTI);
    }
}

void AssimpWrapper::loadTriangles() {
    triangleCount = 0;
    for (int i = 0; i < loadedScene->mNumMeshes; i++) {
        triangleCount += loadedScene->mMeshes[i]->mNumFaces;
    }
    triangles = new Triangle[triangleCount];
    for (int i = 0, k = 0; i < loadedScene->mNumMeshes; i++) {
        aiMesh *mesh = loadedScene->mMeshes[i];
        for (int j = 0; j < loadedScene->mMeshes[i]->mNumFaces; j++) {
            aiFace *face = &mesh->mFaces[j];
            if (face->mNumIndices != 3) {
                throw ParseError("Not a triangle");
            }
            triangles[k].x = getPoint(mesh, face->mIndices[0]);
            triangles[k].y = getPoint(mesh, face->mIndices[1]);
            triangles[k].z = getPoint(mesh, face->mIndices[2]);
            triangles[k].materialCode = mesh->mMaterialIndex;
            if (mesh->HasNormals()) {
                triangles[k].normalVector =
                        Vector::ZERO
                                .add(getNormal(mesh, face->mIndices[0]))
                                .add(getNormal(mesh, face->mIndices[1]))
                                .add(getNormal(mesh, face->mIndices[2])).normalize();
            }
            k++;
        }
    }
}

std::unique_ptr<Scene> AssimpWrapper::load(const std::string &filename) {
    Assimp::Importer importer;

    loadedScene = importer.ReadFile(filename,
                                    aiProcess_Triangulate |
                                    aiProcess_JoinIdenticalVertices);

    if (loadedScene == nullptr) {
        throw ParseError(importer.GetErrorString());
    }
    loadMaterials();
    loadTriangles();

    return std::make_unique<Scene>(Scene(materialsCount, materials, triangleCount, triangles));
}
