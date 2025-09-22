#include "GmshReader.hpp"

int main() {
    std::string fName = "square";
    std::string fPath = "/home/lucas/Documents/GLASs/Tests/GmshReader/";
    GmshReader<uint32_t, float> test(fName, fPath);
    test.readMesh();
}