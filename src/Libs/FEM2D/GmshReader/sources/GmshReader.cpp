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

    // Ensure file format is 2.2
    std::getline(meshFile, fileLine); // This should be $MeshFormat
    if (fileLine != "$MeshFormat") {
        std::cerr << "Error: No $MeshFormat tag found in mesh file!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::getline(meshFile, fileLine); // This should be 2.2 X Y
    // Split the line by spaces
    std::istringstream iss(fileLine);
    std::string version;
    iss >> version;
    if (version != "2.2") {
        std::cerr << "Error: Unsupported mesh file version " << version << "! Only version 2.2 is supported." << std::endl;
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
void GmshReader<ITYPE, RTYPE>::readNodes() {
    std::cout << "--| GmshReader: Reading node data from file..." << std::endl;

    // Go back to beginning of file
    meshFile.clear();
    meshFile.seekg(0, std::ios::beg);

    // Find $Nodes section
    while (std::getline(meshFile, fileLine)) {
        if (fileLine == "$Nodes") {
            // Bypass first line
            std::getline(meshFile, fileLine);
            // Read node data
            for (ITYPE i = 0; i < numPoints; ++i) {
                // Get file line
                std::getline(meshFile, fileLine);
                // Split the line by spaces
                std::istringstream iss(fileLine);
                // Get tokens
                std::string nodeID, xpos, ypos;
                iss >> nodeID >> xpos >> ypos;
                // Store in table
                coordsTable[i][0] = static_cast<RTYPE>(std::stod(xpos));
                coordsTable[i][1] = static_cast<RTYPE>(std::stod(ypos));
            }
            // Read end of section
            std::getline(meshFile, fileLine); // This should be $EndNodes
            if (fileLine != "$EndNodes") {
                std::cerr << "Error: No $EndNodes tag found in mesh file!" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            break;
        }
    }
}

template <typename ITYPE, typename RTYPE>
void GmshReader<ITYPE, RTYPE>::readElements() {
    std::cout << "--| GmshReader: Reading $Elements section from file..." << std::endl;

    // No need to rewind file, pick up right after $EndNodes
    while (std::getline(meshFile, fileLine)) {
        if (fileLine == "$Elements") {
            // Bypass first line
            std::getline(meshFile, fileLine);

            // Read the boundary elements (1st portion of table)
            std::cout << "--| GmshReader: Reading boundary element data from file..." << std::endl;
            for (ITYPE i = 0; i < numBoundElements; ++i) {
                // Get file line
                std::getline(meshFile, fileLine);
                // Split the line by spaces
                std::istringstream iss(fileLine);
                // Get tokens
                std::string elemID, elemType, numTags, tag1, tag2, node1, node2;
                iss >> elemID >> elemType >> numTags >> tag1 >> tag2 >> node1 >> node2;
                // Store in table (subtract 1 from node IDs to convert to 0-based indexing)
                boundElemTable[i][0] = static_cast<ITYPE>(std::stoul(node1)) - 1;
                boundElemTable[i][1] = static_cast<ITYPE>(std::stoul(node2)) - 1;
            }

            // Read the internal elements (2nd portion of table)
            std::cout << "--| GmshReader: Reading internal element data from file..." << std::endl;
            for (ITYPE i = 0; i < numElements; ++i) {
                // Get file line
                std::getline(meshFile, fileLine);
                // Split the line by spaces
                std::istringstream iss(fileLine);
                // Get tokens
                std::string elemID, elemType, numTags, tag1, tag2, node1, node2, node3, node4;
                iss >> elemID >> elemType >> numTags >> tag1 >> tag2 >> node1 >> node2 >> node3 >> node4;
                // Store in table (subtract 1 from node IDs to convert to 0-based indexing)
                elemTable[i][0] = static_cast<ITYPE>(std::stoul(node1)) - 1;
                elemTable[i][1] = static_cast<ITYPE>(std::stoul(node2)) - 1;
                elemTable[i][2] = static_cast<ITYPE>(std::stoul(node3)) - 1;
                elemTable[i][3] = static_cast<ITYPE>(std::stoul(node4)) - 1;
            }
        }
    }
}

template <typename ITYPE, typename RTYPE>
void GmshReader<ITYPE, RTYPE>::readMesh() {
    std::cout << "--| GmshReader: Reading mesh data from file..." << std::endl;

    // Initialize internal data structures
    init();

    // Read nodes
    readNodes();

    // Read elements
    readElements();
}

template class GmshReader<uint32_t, float>;
template class GmshReader<uint64_t, float>;
template class GmshReader<uint32_t, double>;
template class GmshReader<uint64_t, double>;