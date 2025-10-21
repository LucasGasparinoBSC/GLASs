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
    #ifdef NCCL_COMMS
        #include <nccl.h>
		// NCCL cheking macro
		#define NCCL_CHECK(call) \
		    { \
		        ncclResult_t err = call; \
		        if (err != ncclSuccess) { \
		            std::cerr << "NCCL error: " << ncclGetErrorString(err) << " at line " << __LINE__ << std::endl; \
		            std::exit(EXIT_FAILURE); \
		        } \
		    }
    #endif
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
	#ifdef USE_GPU
	    template <> inline MPI_Datatype MPIType<__nv_bfloat16>() {
	        MPI_Datatype mpi_bfloat16_type;
	        MPI_Type_contiguous(2, MPI_BYTE, &mpi_bfloat16_type);
	        MPI_Type_commit(&mpi_bfloat16_type);
	        return mpi_bfloat16_type;
	    }
	#endif
}

// NCCLType helper template
#ifdef NCCL_COMMS
	namespace nccl_utils {
		template <typename T> ncclDataType_t NCCLType();

		// Specific specializations
		template <> inline ncclDataType_t NCCLType<int>() { return ncclInt; }
		template <> inline ncclDataType_t NCCLType<uint32_t>() { return ncclUint32; }
		template <> inline ncclDataType_t NCCLType<uint64_t>() { return ncclUint64; }
		template <> inline ncclDataType_t NCCLType<float>() { return ncclFloat; }
		template <> inline ncclDataType_t NCCLType<double>() { return ncclDouble; }
		template <> inline ncclDataType_t NCCLType<__nv_bfloat16>() { return ncclBfloat16; }
	}
#endif

class Comm_Utils
{
    private:
        // MPI communicator groups
        const MPI_Comm world_comm = MPI_COMM_WORLD; // Global communicator
        MPI_Comm lib_comm;                          // Library communicator, dup of client_comm

        // MPI ranks and sizes
        int world_rank = 0;   // Rank in the global communicator
        int world_size;   // Size of the global communicator
        int lib_rank;     // Rank in the library communicator
        int lib_size;     // Size of the library communicator
		// NCCL variables
		#ifdef NCCL_COMMS
			ncclComm_t nccl_comm;     // NCCL communicator
			ncclUniqueId nccl_uid;    // NCCL unique ID
			ncclResult_t nccl_stat;   // NCCL status
			cudaStream_t nccl_stream; // NCCL CUDA stream
		#endif
    public:
        // Flag for parallel execution
        bool isParallel = false; // Flag for parallel execution

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

        // Allreduce wrappers
        template <typename VTYPE>
        void Allreduce_Sum(VTYPE* sendbuf, VTYPE* recvbuf, int count);
};

#endif