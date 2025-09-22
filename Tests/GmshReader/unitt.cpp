#include "GmshReader.hpp"

int main() {
    std::string fName = "square";
    std::string fPath = "/home/lucas/Documents/GLASs/Tests/GmshReader/";
    GmshReader<uint32_t, float> test(fName, fPath);
    test.readMesh();
    uint32_t nElems = test.getNumElements();
    if (nElems != 16) {
        std::cerr << "Error: number of elements read is incorrect (" << nElems << " != 16)" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint32_t nPoints = test.getNumPoints();
    if (nPoints != 25) {
        std::cerr << "Error: number of points read is incorrect (" << nPoints << " != 25)" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint32_t nBoundElems = test.getNumBoundElements();
    if (nBoundElems != 16) {
        std::cerr << "Error: number of boundary elements read is incorrect (" << nBoundElems << " != 16)" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint32_t** testPtrInt = test.getElemTable();
    if (testPtrInt == nullptr) {
        std::cerr << "Error: element table pointer is null" << std::endl;
        exit(EXIT_FAILURE);
    }
    testPtrInt = test.getBoundElemTable();
    if (testPtrInt == nullptr)
    {
        std::cerr << "Error: element table pointer is null" << std::endl;
        exit(EXIT_FAILURE);
    }
    float** testPtrFloat = test.getCoordsTable();
    if (testPtrFloat == nullptr) {
        std::cerr << "Error: coordinates table pointer is null" << std::endl;
        exit(EXIT_FAILURE);
    }
}