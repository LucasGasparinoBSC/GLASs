#ifndef GMSHREADER_HPP
#define GMSHREADER_HPP

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <cstdlib>

template <typename ITYPE, typename RTYPE>
class GmshReader
{
    private:
        // Base quantities
        const ITYPE numDims = 2;
        const ITYPE numElemNodes = 4;
        const ITYPE numBoundElemNodes = 2;
        ITYPE numPoints;
        ITYPE numBoundPoints;
        ITYPE numPeriodicPoints;
        ITYPE numElements;
        ITYPE numBoundElements;
        // Tables
        ITYPE** elemTable;
        ITYPE** boundElemTable;
        RTYPE** coordsTable;
        // TODO: add tables for periodicity shit
        // ...
        // File operation vars
        std::ifstream meshFile;
        std::string fileName;
        std::string filePath;
        std::string fileLine;
        const std::string fileExt = ".msh";

        // Private methods:
        void init();
        void readNodes();
        void readElements();
    public:
        // Constructor
        GmshReader(const std::string& name, const std::string& path);

        // Destructor
        ~GmshReader();

        // Getters
        ITYPE getNumDims() const { return numDims; };
        ITYPE getNumPoints() const { return numPoints; };
        ITYPE getNumBoundPoints() const { return numBoundPoints; };
        ITYPE getNumPeriodicPoints() const { return numPeriodicPoints; };
        ITYPE getNumElements() const { return numElements; };
        ITYPE getNumBoundElements() const { return numBoundElements; };
        ITYPE getNumElemNodes() const { return numElemNodes; };
        ITYPE getNumBoundElemNodes() const { return numBoundElemNodes; };
        ITYPE** getElemTable() const { return elemTable; };
        ITYPE** getBoundElemTable() const { return boundElemTable; };
        RTYPE** getCoordsTable() const { return coordsTable; };

        void readMesh();
};

#endif //! GMSHREADER_HPP