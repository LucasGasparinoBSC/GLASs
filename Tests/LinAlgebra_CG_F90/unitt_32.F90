module mod_laplacian
   use iso_c_binding
   use cg_wrapper_mod
   implicit none

   type FDM1D_t
      integer(c_int32_t)     :: ncoeff=0           ! Number of coefficients
      integer(c_int32_t)     :: half_stencil=0     ! Half stencil width (number of coefficients on one side of the diagonal)
      integer(c_int32_t)     :: ndof=0             ! Number of degrees of freedom (excludes periodic node at end)
      real(c_float)          :: dx                 ! Spacing between nodes
      real(c_float), pointer :: coeff(:) => null() ! Coefficients for the finite difference stencil
      real(c_float), pointer :: sigma(:) => null() ! Linear term coefficient

      contains
         procedure, pass :: matvec  => FDM1D_matvec  ! Abstract matvec operator
         procedure, pass :: precond => FDM1D_precond ! Abstract precond operator
   end type

   public :: FDM1D_matvec_c, FDM1D_precond_c

   contains

      subroutine FDM1D_matvec_host(this, x_in, x_out)
         implicit none
         class(FDM1D_t), intent(inout) :: this
         real(c_float),   intent(in)   :: x_in(this%ndof)
         real(c_float),   intent(out)  :: x_out(this%ndof)
         integer(c_int32_t)            :: idxStart, idxEnd ! Extremes of the domain where the stencil can be applied without going out of bounds
         integer(c_int32_t)            :: idx, j, idx_w
         real(8)                       :: tmp

         ! Zero x_out
         x_out(:) = 0.0_c_float

         idxStart = this%half_stencil + 1
         idxEnd   = this%ndof - this%half_stencil

         ! Non-periodic stencil application
         do idx = idxStart, idxEnd
            tmp = 0.0d0
            do j = -this%half_stencil, this%half_stencil
               tmp = tmp + real( this%coeff(j+this%half_stencil+1) * x_in(idx+j), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + this%sigma(idx) * x_in(idx)
         end do

         ! Left bc
         do idx = 1, idxStart-1
            tmp = 0.0d0
            do j = -this%half_stencil, this%half_stencil
               ! Compute wrapped index for periodic boundary conditions
               idx_w = modulo(idx+j-1, this%ndof) + 1
               tmp = tmp + real( this%coeff(j+this%half_stencil+1) * x_in(idx_w), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + this%sigma(idx) * x_in(idx)
         end do

         ! right bc
         do idx = idxEnd+1, this%ndof
            tmp = 0.0d0
            do j = -this%half_stencil, this%half_stencil
               ! Compute wrapped index for periodic boundary conditions
               idx_w = modulo(idx+j-1, this%ndof) + 1
               tmp = tmp + real( this%coeff(j+this%half_stencil+1) * x_in(idx_w), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + this%sigma(idx) * x_in(idx)
         end do

      end subroutine FDM1D_matvec_host

      subroutine FDM1D_matvec(this, x_in, x_out)
         implicit none
         class(FDM1D_t), intent(inout) :: this
         real(c_float),   intent(in)   :: x_in(this%ndof)
         real(c_float),   intent(out)  :: x_out(this%ndof)
         integer(c_int32_t)            :: idxStart, idxEnd ! Extremes of the domain where the stencil can be applied without going out of bounds
         integer(c_int32_t)            :: idx, j, idx_w
         integer(c_int32_t)            :: ndof, hs
         real(c_float), pointer        :: coeff(:), sigma(:)
         real(8)                       :: tmp

         hs = this%half_stencil
         ndof = this%ndof
         coeff => this%coeff
         sigma => this%sigma

         ! Zero x_out
         !$acc kernels deviceptr(x_out)
         x_out(:) = 0.0_c_float
         !$acc end kernels

         idxStart = hs + 1
         idxEnd   = ndof - hs

         ! Non-periodic stencil application
         !$acc parallel loop present(coeff, sigma) deviceptr(x_in, x_out) private(tmp) async(1)
         do idx = idxStart, idxEnd
            tmp = 0.0d0
            !$acc loop seq
            do j = -hs, hs
               tmp = tmp + real( coeff(j+hs+1) * x_in(idx+j), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + sigma(idx) * x_in(idx)
         end do
         !$acc end parallel loop

         ! Left bc
         !$acc parallel loop present(coeff, sigma) deviceptr(x_in, x_out) private(tmp, idx_w) async(2)
         do idx = 1, idxStart-1
            tmp = 0.0d0
            !$acc loop seq
            do j = -hs, hs
               ! Compute wrapped index for periodic boundary conditions
               idx_w = modulo(idx+j-1, ndof) + 1
               tmp = tmp + real( coeff(j+hs+1) * x_in(idx_w), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + sigma(idx) * x_in(idx)
         end do
         !$acc end parallel loop

         ! right bc
         !$acc parallel loop present(coeff, sigma) deviceptr(x_in, x_out) private(tmp, idx_w) async(3)
         do idx = idxEnd+1, ndof
            tmp = 0.0d0
            !$acc loop seq
            do j = -hs, hs
               ! Compute wrapped index for periodic boundary conditions
               idx_w = modulo(idx+j-1, ndof) + 1
               tmp = tmp + real( coeff(j+hs+1) * x_in(idx_w), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + sigma(idx) * x_in(idx)
         end do
         !$acc end parallel loop

         !$acc wait(1,2,3)

      end subroutine FDM1D_matvec

      ! Simple diagonal preconditioning
      subroutine FDM1D_precond(this, x_in, x_out)
         implicit none
         class(FDM1D_t), intent(inout) :: this
         real(c_float),   intent(in)   :: x_in(this%ndof)
         real(c_float),   intent(out)  :: x_out(this%ndof)
         integer(c_int32_t)            :: i, ndof, ncoeff
         real(c_float)                 :: diagcoeff
         real(c_float), pointer        :: coeff(:), sigma(:)

         ndof  = this%ndof
         ncoeff = this%ncoeff
         coeff => this%coeff
         sigma => this%sigma

         !$acc parallel loop present(coeff, sigma) deviceptr(x_in, x_out) private(diagcoeff)
         do i = 1, ndof
            diagcoeff = 1.0/( coeff( (ncoeff+1)/2 ) + sigma(i) ) ! Diagonal coefficient is always the middle coefficient for a symmetric stencil, inverted
            x_out(i) = x_in(i) * diagcoeff
         end do
         !$acc end parallel loop
      end subroutine FDM1D_precond

      ! C wrapper for the Fortran matvec
      subroutine FDM1D_matvec_c(x_in, x_out, opData) bind(C)
         implicit none
         real(c_float), intent(in)  :: x_in(*)
         real(c_float), intent(out) :: x_out(*)
         type(c_ptr),   value       :: opData
         type(FDM1D_t), pointer     :: fdmData

         call c_f_pointer(opData, fdmData)
         call fdmData%matvec(x_in, x_out)
      end subroutine FDM1D_matvec_c

      ! C wrapper for the Fortran preconditioner
      subroutine FDM1D_precond_c(x_in, x_out, opData) bind(C)
         implicit none
         real(c_float), intent(in)  :: x_in(*)
         real(c_float), intent(out) :: x_out(*)
         type(c_ptr),   value       :: opData
         type(FDM1D_t), pointer     :: fdmData

         call c_f_pointer(opData, fdmData)
         call fdmData%precond(x_in, x_out)
      end subroutine FDM1D_precond_c

end module mod_laplacian

program test_32
   use mpi
   use iso_c_binding
   use cg_wrapper_mod
   use mod_laplacian
   implicit none

   ! MPI vars
   integer :: ierr, irank, nranks, client_comm

   ! Basic data
   integer(c_int32_t), parameter :: nNodes = 2000001
   integer(c_int32_t), parameter :: maxIters = 1000
   integer(c_int32_t), parameter :: pOrder = 4
   integer(c_int32_t), parameter :: nruns = 20
   real(c_double)    , parameter :: tol = 1.0e-7_c_double
   real(c_float)     , parameter :: pi = 3.14159265358979323846_c_float

   ! Internal vars
   integer(c_int32_t)              :: nWorking, i, j, k, nListEntries
   integer(c_int32_t), allocatable :: listEntries(:)
   real(c_float)                   :: err, max_err
   real(c_float), allocatable      :: gridPts(:), x0(:), rhs(:), x_solve(:), Axsolve(:), x_exact(:)
   type(FDM1D_t), target           :: laplObj
   type(c_ptr)                     :: glassSolver
   type(c_ptr)                     :: opData
   type(c_funptr)                  :: matvecFunc
   type(c_funptr)                  :: precondFunc

   ! Initialize MPI
   call MPI_Init(ierr)

   ! Client communicator is world comm
   client_comm = MPI_COMM_WORLD
   call MPI_Comm_rank(client_comm, irank, ierr)
   call MPI_Comm_size(client_comm, nranks, ierr)

   ! For now, crash if more than one rank is used
   if (nranks > 1) then
      if (irank == 0) then
         print *, "This test is only for single rank execution"
      end if
      call MPI_Abort(client_comm, 1, ierr)
   end if

   ! Working nodes is the number of nodes minus 1
   nWorking = nNodes - 1
   nListEntries = nWorking
   allocate(listEntries(nListEntries))
   !$acc enter data create(listEntries)
   !$acc parallel loop present(listEntries)
   do i = 1, nListEntries
      listEntries(i) = i-1
   end do
   !$acc end parallel loop

   ! Set object info
   laplObj%ndof         = nWorking
   laplObj%ncoeff       = pOrder + 1
   laplObj%half_stencil = pOrder / 2
   laplObj%dx           = (pi * 2.0_c_float) / real(nWorking, c_float)
   allocate(laplObj%coeff(laplObj%ncoeff))
   allocate(laplObj%sigma(laplObj%ndof), source=0.0_c_float)
   if (pOrder == 2) then
      laplObj%coeff = -[1.0_c_float, -2.0_c_float, 1.0_c_float] / (laplObj%dx**2)
   else if (pOrder == 4) then
      laplObj%coeff = -[-1.0_c_float/12.0_c_float, 4.0_c_float/3.0_c_float, -5.0_c_float/2.0_c_float, 4.0_c_float/3.0_c_float, -1.0_c_float/12.0_c_float] / (laplObj%dx**2)
   else
      print *, "Unsupported order"
      call MPI_Abort(client_comm, 1, ierr)
   end if
   !$acc enter data copyin(laplObj%coeff, laplObj%sigma)
   !$acc enter data copyin(laplObj)
   !$acc enter data attach(laplObj%coeff, laplObj%sigma)

   ! Create grid points x0 and rhs
   ! x0 is going to be randomly initialized with values between -1 and 1, rhs is going to be sin(x) evaluated at the grid points
   allocate(gridPts(nWorking), rhs(nWorking), x0(nWorking), x_exact(nWorking))
   !$acc enter data create(x0, rhs, x_exact)
   do i = 1, nWorking
      gridPts(i) = real( (i-1), c_float ) * laplObj%dx
      x_exact(i) = sin(gridPts(i))
      call random_number(x0(i))
      x0(i) = (2.0_c_float * x0(i)) - 1.0_c_float
      laplObj%sigma(i) = 1.0_c_float / (laplObj%dx**2)
   end do
   laplObj%sigma(nWorking/2) = 100.0_c_float / (laplObj%dx**2)
   call FDM1D_matvec_host(laplObj, x_exact, rhs)
   !$acc update device(x0, rhs, x_exact, laplObj%sigma)

   ! Create the GLASs solver
   glassSolver = cg_create_u32_pf(client_comm, nWorking, nListEntries, maxIters, tol)

   ! Setup x0 and b
   !$acc host_data use_device(listEntries, x0, rhs)
   call cg_setup_u32_f(glassSolver, listEntries, x0, rhs)
   !$acc end host_data

   ! Setup the matvec and preconditioner
   opData = c_loc(laplObj)
   matvecFunc = c_funloc(FDM1D_matvec_c)
   precondFunc = c_funloc(FDM1D_precond_c)

   ! Call the FPCG solver a couple of times
   do k = 1, nruns
      call fpcg_solve_u32_f(glassSolver, matvecFunc, precondFunc, opData)
      !call cg_solve_u32_f(glassSolver, matvecFunc, opData)
   end do

   ! Get the solution back
   allocate(x_solve(nWorking))
   !$acc enter data create(x_solve)
   !$acc host_data use_device(x_solve)
   call cg_get_solution_u32_f(glassSolver, x_solve)
   !$acc end host_data
   !$acc update host(x_solve)

   ! 3. HARD VERIFICATION: Compare x_solve directly back to x_exact
   max_err = 0.0_c_float
   do i = 1, nWorking
      err = abs(x_solve(i) - x_exact(i))
      if (err > max_err) max_err = err
      
      ! Single precision floor safety bound check
      if (err > 1.0e-4_c_float) then
         print *, "Rank ", irank, ": Verification failed at index ", i
         print *, "Expected: ", x_exact(i), " Got: ", x_solve(i), " Delta: ", err
         call MPI_Abort(client_comm, 1, ierr)
      end if
   end do

   if (irank == 0) then
      print *, "Verification Passed! Discrete equation matches perfectly."
      print *, "Maximum absolute solution error: ", max_err
   end if

   ! Finalize MPI
   call MPI_Finalize(ierr)
end program test_32