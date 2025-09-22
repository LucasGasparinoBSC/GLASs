#include "GmshReader.hpp"

template <typename ITYPE, typename RTYPE>
GmshReader<ITYPE, RTYPE>::GmshReader(const std::string& name, const std::string& path) {
    std::cout << "--| GmshReader: Initializing reader for file " << name + fileExt << " at path " << path << std::endl;

    // Initialize all base quantities to zero
    numPoints = 0;
    numBoundPoints = 0;
    numPeriodicPoints = 0;
    numElements = 0;
    numBoundElements = 0;

    // Initialize all tables to nullptr
    elemTable = nullptr;
    boundElemTable = nullptr;
    coordsTable = nullptr;

    // Set file name and path
    fileName = name;
    filePath = path;
    // Open the file for reading (ASCII format only)
    meshFile.open(filePath + fileName + fileExt);
    if (!meshFile.is_open()) {
        std::cerr << "Error: Could not open mesh file " << filePath + fileName + fileExt << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "--| GmshReader: reader object ready for use!" << std::endl;
}

template <typename ITYPE, typename RTYPE>
GmshReader<ITYPE, RTYPE>::~GmshReader() {
}

template <typename ITYPE, typename RTYPE>
void GmshReader<ITYPE, RTYPE>::init() {
    std::cout << "--| GmshReader: Initializing internal data structures..." << std::endl;
    // Parse the file and extract basic info from Nodes and Elements
    while (std::getline(meshFile, fileLine)) {
        // $Nodes section
        if (fileLine == "$Nodes") {
            // Next line is the number of nodes
            std::getline(meshFile, fileLine);
            numPoints = static_cast<ITYPE>(std::stoul(fileLine));
        }
        // $Elements section
        if (fileLine == "$Elements") {
            // Next line is total number of elements
            std::getline(meshFile, fileLine);
            ITYPE totalElements = static_cast<ITYPE>(std::stoul(fileLine));
            // Loop through table to count elements by type
            ITYPE elemType = 0;
            ITYPE count = 0;
            for (ITYPE i = 0; i < totalElements; ++i) {
                std::getline(meshFile, fileLine);
                // Split the line by spaces
                std::istringstream iss(fileLine);
                // Get 2nd token
                std::string tk1, tk2;
                iss >> tk1 >> tk2;
                elemType = static_cast<ITYPE>(std::stoul(tk2));
                if (elemType == 1) count++;
            }
            numBoundElements = count;
            numElements = totalElements - numBoundElements;
            // Break out after this section
            break;
        }
    }
    std::cout << "--| GmshReader: Found " << numPoints << " points, " << numElements << " elements and " << numBoundElements << " boundary elements." << std::endl;

    std::cout << "--| GmshReader: Allocating internal data structures..." << std::endl;
    // Allocate coords table
    coordsTable = (RTYPE**)calloc(numPoints, sizeof(RTYPE*));
    for (ITYPE i = 0; i < numPoints; ++i) {
        coordsTable[i] = (RTYPE*)calloc(numDims, sizeof(RTYPE));
    }

    // Allocate elem table
    elemTable = (ITYPE**)calloc(numElements, sizeof(ITYPE*));
    for (ITYPE i = 0; i < numElements; ++i) {
        elemTable[i] = (ITYPE*)calloc(numElemNodes, sizeof(ITYPE));
    }

    // Allocate bound elem table
    boundElemTable = (ITYPE**)calloc(numBoundElements, sizeof(ITYPE*));
    for (ITYPE i = 0; i < numBoundElements; ++i) {
        boundElemTable[i] = (ITYPE*)calloc(numBoundElemNodes, sizeof(ITYPE));
    }
    std::cout << "--| GmshReader: Internal data structures allocated!" << std::endl;
}

template <typename ITYPE, typename RTYPE>
void GmshReader<ITYPE, RTYPE>::readMesh() {
    std::cout << "--| GmshReader: Reading mesh data from file..." << std::endl;
    // Initialize internal data structures
    init();
}

template class GmshReader<uint32_t, float>;
template class GmshReader<uint64_t, float>;
template class GmshReader<uint32_t, double>;
template class GmshReader<uint64_t, double>;