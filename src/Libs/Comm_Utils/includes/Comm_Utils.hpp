#ifndef COMM_UTILS_H
#define COMM_UTILS_H

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <mpi.h>
#ifdef USE_GPU
#include "CUDA_Utils.cuh"
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

// Macro for checking MPI errors
#define MPI_CHECK(call) \
    { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char err_string[256]; \
            int err_length; \
            MPI_Error_string(err, err_string, &err_length); \
            std::cerr << "MPI error: " << err_string << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    }

// MPIType helper template
namespace mpi_utils {
    template <typename T> MPI_Datatype MPIType();

    // Specific specializations
    template <> inline MPI_Datatype MPIType<int>() { return MPI_INT; }
    template <> inline MPI_Datatype MPIType<uint32_t>() { return MPI_UINT32_T; }
    template <> inline MPI_Datatype MPIType<uint64_t>() { return MPI_UINT64_T; }
    template <> inline MPI_Datatype MPIType<float>() { return MPI_FLOAT; }
    template <> inline MPI_Datatype MPIType<double>() { return MPI_DOUBLE; }
    // TODO: add support for nv_bfloat16
}

template <typename ITYPE, typename RTYPE>
class Comm_Utils
{
    private:
        // MPI communicator groups
        const MPI_Comm world_comm = MPI_COMM_WORLD; // Global communicator
        MPI_Comm lib_comm;                          // Library communicator, dup of client_comm

        // MPI ranks and sizes
        int world_rank;   // Rank in the global communicator
        int world_size;   // Size of the global communicator
        int lib_rank;     // Rank in the library communicator
        int lib_size;     // Size of the library communicator
    public:
        // Empty constructor
        Comm_Utils() {};

        // Constructor
        Comm_Utils(MPI_Comm& client_comm);

        // Destructor
        ~Comm_Utils();

        // Setup method
        void setup(MPI_Comm& client_comm);

        // Getters
        int getWorldRank() { return this->world_rank; };
        int getWorldSize() { return this->world_size; };
        int getLibRank() { return this->lib_rank; };
        int getLibSize() { return this->lib_size; };
        MPI_Comm getLibComm() { return this->lib_comm; };
};

#endif