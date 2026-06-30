#ifndef COMM_UTILS_HPP
#define COMM_UTILS_HPP

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
    #include "DeviceMemory.hpp"
    #include "Kernels_Lv1.cuh" // Check if needed
    #ifdef NCCL_COMMS
        #include <nccl.h>
		// NCCL cheking macro
		#define NCCL_CHECK(call) \
            do                                               \
		    {                                                \
		        ncclResult_t _err = (call);                  \
		        if (_err != ncclSuccess) {                   \
		            std::cerr << "NCCL error: "              \
                              << ncclGetErrorString(_err)    \
                              << " (" << _err << ") at "     \
                              << __FILE__ << ":" << __LINE__ \
                              << std::endl;                  \
		            std::exit(EXIT_FAILURE);                 \
		        }                                            \
		    } while (0)
    #elif defined(RCCL_COMMS)
        #include <rccl.h>
        // RCCL checking macro
    #endif
#else
    // If using CPU mode, PUSH_RANGE and POP_RANGE do nothing
    // This will propagate to the rest of the libraries
    #define PUSH_RANGE(name,cid)
    #define POP_RANGE()
#endif

#define MPI_CHECK(call)                                        \
    do                                                         \
    {                                                          \
        int _err = (call);                                     \
        if (_err != MPI_SUCCESS)                               \
        {                                                      \
            int _rank = -1;                                    \
            MPI_Comm_rank(MPI_COMM_WORLD, &_rank);             \
            char err_string[MPI_MAX_ERROR_STRING];             \
            int err_length = 0;                                \
            MPI_Error_string(_err, err_string, &err_length);   \
            std::cerr << "[Rank " << _rank << "] MPI error: "  \
                      << std::string(err_string, err_length)   \
                      << " in " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                            \
            MPI_Abort(MPI_COMM_WORLD, _err);                   \
        }                                                      \
    } while (0)

// MPIType helper template
namespace mpi_utils {
    template <typename T> MPI_Datatype MPIType();

    // Specific specializations
    template <> inline MPI_Datatype MPIType<int>() { return MPI_INT; }
    template <> inline MPI_Datatype MPIType<uint32_t>() { return MPI_UINT32_T; }
    template <> inline MPI_Datatype MPIType<uint64_t>() { return MPI_UINT64_T; }
    template <> inline MPI_Datatype MPIType<float>() { return MPI_FLOAT; }
    template <> inline MPI_Datatype MPIType<double>() { return MPI_DOUBLE; }
    // TODO: add support for bf16
    // NOTE: in principle, NCCL should be used for GPU2GPU comms, so a dummy bf16 MPI type is defined here
	#ifdef USE_GPU
	    template <> inline MPI_Datatype MPIType<DeviceUtils::bf16>() {
	        MPI_Datatype mpi_bfloat16_type;
	        MPI_Type_contiguous(2, MPI_BYTE, &mpi_bfloat16_type);
	        MPI_Type_commit(&mpi_bfloat16_type);
	        return mpi_bfloat16_type;
	    }
	#endif
}

#ifdef NCCL_COMMS
    // NCCLType helper template
	namespace nccl_utils {
		template <typename T> ncclDataType_t NCCLType();

		// Specific specializations
		template <> inline ncclDataType_t NCCLType<int>() { return ncclInt; }
		template <> inline ncclDataType_t NCCLType<uint32_t>() { return ncclUint32; }
		template <> inline ncclDataType_t NCCLType<uint64_t>() { return ncclUint64; }
		template <> inline ncclDataType_t NCCLType<float>() { return ncclFloat; }
		template <> inline ncclDataType_t NCCLType<double>() { return ncclDouble; }
		template <> inline ncclDataType_t NCCLType<DeviceUtils::bf16>() { return ncclBfloat16; }
	}
#elif defined(RCCL_COMMS)
    // RCCLType helper template
    namespace rccl_utils {
        template <typename T> rcclDataType_t RCCLType();

        // Specific specializations
        template <> inline rcclDataType_t RCCLType<int>() { return rcclInt; }
        template <> inline rcclDataType_t RCCLType<uint32_t>() { return rcclUint32; }
        template <> inline rcclDataType_t RCCLType<uint64_t>() { return rcclUint64; }
        template <> inline rcclDataType_t RCCLType<float>() { return rcclFloat; }
        template <> inline rcclDataType_t RCCLType<double>() { return rcclDouble; }
        template <> inline rcclDataType_t RCCLType<DeviceUtils::bf16>() { return rcclBfloat16; } // TODO: proper fp16 for AMD
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

		#ifdef NCCL_COMMS
		    // NCCL variables
			ncclComm_t nccl_comm;              // NCCL communicator
			ncclUniqueId nccl_uid;             // NCCL unique ID
			ncclResult_t nccl_stat;            // NCCL status
			DeviceUtils::Stream_t nccl_stream; // NCCL CUDA stream
        #elif defined(RCCL_COMMS)
            // RCCL variables
            // TODO: add RCCL variable set
		#endif
    public:
        // Flag for parallel execution
        bool isParallel = false; // Flag for parallel execution

        // Empty constructor
        Comm_Utils();

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

        // Timing utility
        template <typename FUNC>
        double timeFunction(FUNC&& f) {
            double start = MPI_Wtime();
            f();
            double end = MPI_Wtime();
            double time_local = end - start;
            if (this->lib_size > 1) {
                double time_global = 0.0;
                MPI_Reduce(&time_local, &time_global, 1, MPI_DOUBLE, MPI_MAX, 0, this->lib_comm);
                return time_global;
            } else {
                return time_local;
            }
        }
};

#endif //! COMM_UTILS_HPP