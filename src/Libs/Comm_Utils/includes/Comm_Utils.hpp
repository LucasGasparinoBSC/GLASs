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

template <typename ITYPE, typename RTYPE>
class Comm_Utils
{
    private:
        // MPI communicator groups
        const MPI_Comm world_comm = MPI_COMM_WORLD; // Global communicator
        MPI_Comm client_comm;                       // Client communicator
        MPI_Comm lib_comm;                          // Library communicator, dup of client_comm

        // MPI ranks and sizes
        int world_rank;   // Rank in the global communicator
        int world_size;   // Size of the global communicator
        int client_rank;  // Rank in the client communicator
        int client_size;  // Size of the client communicator
        int lib_rank;     // Rank in the library communicator
        int lib_size;     // Size of the library communicator
    public:
        // Empty constructor
        Comm_Utils() {};

        // Constructor
        Comm_Utils(MPI_Comm comm, ITYPE wr, ITYPE ws, ITYPE cr, ITYPE cs);

        // Destructor
        ~Comm_Utils();

        // Setup method
        void setup(MPI_Comm comm, ITYPE wr, ITYPE ws, ITYPE cr, ITYPE cs);
};

#endif