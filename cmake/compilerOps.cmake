message("-- Selecting compiler Ops...")

# Set library types to be shared
set(LIBRARY_TYPE SHARED)

# Set a default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set C/C++ standard
set(CMAKE_CXX_STANDARD 20)

# Define common C compiler flags
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "")

# Define common CXX compiler flags
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "")

# Define common Fortran compiler flags
set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS})
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0")
set(CMAKE_Fortran_FLAGS_RELEASE "")

# If GPU usage is enabled, define CUDA info and NVCC flags
if(USE_GPU)
	if(USE_NVIDIA_GPU)
		# Get the CUDA architecture and version
		set_cc()
		set_cuda()
		message("-- CUDA architecture: " ${GPU_CC})
		message("-- CUDA version: " ${GPU_CUDA})

		# Set the CUDA_ARCH variable
		set(CUDA_ARCHITECTURES ${GPU_CC})
		set(CMAKE_CUDA_ARCHITECTURES ${GPU_CC})

		# Define CUDA compiler flags
		set(CMAKE_CUDA_FLAGS "-m64 -res-usage --extended-lambda")
		set(CMAKE_CUDA_FLAGS_DEBUG "-pg -g -G -O0 -Wreorder")
		set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
	elseif(USE_AMD_GPU)
		# Set HIP target architecture (user can override)
		# TODO: make this more automagic
		if(NOT DEFINED HIP_TARGET)
			# Default target; adjust based on your hardware
			set(HIP_TARGET gfx908 CACHE STRING "Target AMD GPU architecture")
			message("-- No HIP_TARGET specified. Using default: " ${HIP_TARGET})
		endif()
		message("-- HIP target architecture: " ${HIP_TARGET})

		# Set the HIP_ARCH variable
		set(CMAKE_HIP_ARCHITECTURES ${HIP_TARGET})

		# Define HIP compiler flags
    	set(CMAKE_HIP_FLAGS "-O3 --offload-arch=${HIP_TARGET}")
    	set(CMAKE_HIP_FLAGS_DEBUG "-g -O0")
    	set(CMAKE_HIP_FLAGS_RELEASE "-O3")
	endif()
endif()

# Define specific compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
	message("-- GNU compiler detected")
	if (USE_GPU)
		message(FATAL_ERROR "GPU not supported with GNU compiler")
	endif()
	# Common GNU+MPI flags
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-cpp")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-cpp")
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-cpp -ffree-line-length-none -std=f2008")
	# Debug
	set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic")
	set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic -fbacktrace -Wconversion-extra -ftrapv -fcheck=all -ffpe-trap=invalid,zero,overflow")
	# Release
	set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-O3 -march=native")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-O3 -march=native")
	set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-O3 -march=native")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel" OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
	message("-- Intel compiler detected")
	if (USE_GPU)
		message(FATAL_ERROR "GPU not supported with Intel compiler")
	endif()
	# Common Intel+MPI flags
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "")
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-fpp")
	# Debug
	set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-debug all")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-debug all")
	set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-traceback -ftrapuv -debug all -fpe3 -fpe-all=3")
	# Release
	set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-O3 -qmkl -ipo -qopt-report=3 -mtune=native")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-O3 -qmkl -ipo -qopt-report=3 -mtune=native")
	set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-O3 -qmkl -ipo -qopt-report=3 -mtune=native")
elseif(CMAKE_C_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_C_COMPILER_ID STREQUAL "PGI")
	message("-- NVHPC compiler detected")
	# Based on Fortran compiler version, set the nvtx lib name
	if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 25.1)
		set(NVTX_LIB_F "nvToolsExt")
	else()
		set(NVTX_LIB_F "nvtx3interop")
	endif()
	message("-- NVTX Fortran library: " ${NVTX_LIB_F})
	# Common NVHPC+MPI flags
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-cpp -lstdc++")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-cpp -lstdc++")
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-cpp -lstdc++")
	# Debug
	set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-Minform=inform -C -traceback -Ktrap=fp")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-Minform=inform -C -Mchkstk -traceback -Ktrap=fp")
	set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-Minform=inform -C -Mchkstk -traceback -Ktrap=fp")
	# Release
	set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-fast -mcpu=native")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-fast -mcpu=native")
	set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-fast -mcpu=native")
	# GPU options
	if(USE_GPU)
		if(USE_NCCL)
			message("-- Enabling NCCL support")
			set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-DNCCL_COMMS -cudalib=nccl")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-DNCCL_COMMS -cudalib=nccl")
			set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-DNCCL_COMMS -cudalib=nccl")
		endif()
		# Automatically detect compute capability and CUDA version
		set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-acc -cuda -gpu=cc${GPU_CC},cuda${GPU_CUDA},lineinfo,nordc -Minfo=accel -DUSE_GPU -DUSE_CUDA")
		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-acc -cuda -gpu=cc${GPU_CC},cuda${GPU_CUDA},lineinfo,nordc -Minfo=accel -DUSE_GPU -DUSE_CUDA")
		set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-acc -cuda -gpu=cc${GPU_CC},cuda${GPU_CUDA},lineinfo,nordc -Minfo=accel -DUSE_GPU -DUSE_CUDA -l${NVTX_LIB_F}")
	endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
	message("-- Clang compiler detected")
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-cpp")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-cpp")
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-cpp -ffree-line-length-none -std=f2008")
	# Debug
	set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic")
	set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic -fbacktrace -Wconversion-extra -ftrapv -fcheck=all -ffpe-trap=invalid,zero,overflow")
	# Release
	set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-O3 -mcpu=native")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-O3 -mcpu=native")
	set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-O3 -mcpu=native")
	if(USE_GPU)
		if(USE_AMD_GPU)
			set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-DUSE_GPU -DUSE_HIP")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-DUSE_GPU -DUSE_HIP")
			set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-DUSE_GPU -DUSE_HIP")
		else()
			message(FATAL_ERROR "GPU support for Clang only with AMD GPUs")
		endif()
	endif()
else()
	message("this shit: " ${CMAKE_C_COMPILER_ID})
	message(FATAL_ERROR "Unknown compiler")
endif()

# Adjust stringg so ; is removed from the command
string(REPLACE ";" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE ";" " " CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")
string(REPLACE ";" " " CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE ";" " " CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG}")
string(REPLACE ";" " " CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE ";" " " CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE}")

# Print a summary of the compiler options based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	message("-- Debug build detected")
	message("-- C flags: " ${CMAKE_C_FLAGS} " " ${CMAKE_C_FLAGS_DEBUG})
	message("-- CXX flags: " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_DEBUG})
	message("-- Fortran flags: " ${CMAKE_Fortran_FLAGS} " " ${CMAKE_Fortran_FLAGS_DEBUG})
	if(USE_GPU)
		if(USE_NVIDIA_GPU)
			message("-- CUDA flags: " ${CMAKE_CUDA_FLAGS} " " ${CMAKE_CUDA_FLAGS_DEBUG})
		elseif(USE_AMD_GPU)
			message("-- HIP flags: " ${CMAKE_HIP_FLAGS} " " ${CMAKE_HIP_FLAGS_DEBUG})
		endif()
	endif()
else()
	message("-- Release build detected")
	message("-- C flags: " ${CMAKE_C_FLAGS} " " ${CMAKE_C_FLAGS_RELEASE})
	message("-- CXX flags: " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_RELEASE})
	message("-- Fortran flags: " ${CMAKE_Fortran_FLAGS} " " ${CMAKE_Fortran_FLAGS_RELEASE})
	if(USE_GPU)
		if(USE_NVIDIA_GPU)
			message("-- CUDA flags: " ${CMAKE_CUDA_FLAGS} " " ${CMAKE_CUDA_FLAGS_RELEASE})
		elseif(USE_AMD_GPU)
			message("-- HIP flags: " ${CMAKE_HIP_FLAGS} " " ${CMAKE_HIP_FLAGS_RELEASE})
		endif()
	endif()
endif()