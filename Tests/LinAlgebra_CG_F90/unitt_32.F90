module mod_laplacian
   use iso_c_binding
   use cg_wrapper_mod
   implicit none

   type FDM1D_t
      integer(c_int32_t)     :: ncoeff=0           ! Number of coefficients
      integer(c_int32_t)     :: half_stencil=0     ! Half stencil width (number of coefficients on one side of the diagonal)
      integer(c_int32_t)     :: ndof=0             ! Number of degrees of freedom (excludes periodic node at end)
      real(c_float), pointer :: coeff(:) => null() ! Coefficients for the finite difference stencil
      real(c_float)          :: dx                 ! Spacing between nodes
      real(c_float)          :: sigma              ! Linear term coefficient

      contains
         procedure, pass :: matvec  => FDM1D_matvec  ! Abstract matvec operator
         procedure, pass :: precond => FDM1D_precond ! Abstract precond operator
   end type

   public :: FDM1D_matvec_c, FDM1D_precond_c

   contains

      subroutine FDM1D_matvec(this, x_in, x_out)
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
            x_out(idx) = real(tmp, c_float) + this%sigma * x_in(idx)
         end do

         ! Left bc
         do idx = 1, idxStart-1
            tmp = 0.0d0
            do j = -this%half_stencil, this%half_stencil
               ! Compute wrapped index for periodic boundary conditions
               idx_w = modulo(idx+j-1, this%ndof) + 1
               tmp = tmp + real( this%coeff(j+this%half_stencil+1) * x_in(idx_w), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + this%sigma * x_in(idx)
         end do

         ! right bc
         do idx = idxEnd+1, this%ndof
            tmp = 0.0d0
            do j = -this%half_stencil, this%half_stencil
               ! Compute wrapped index for periodic boundary conditions
               idx_w = modulo(idx+j-1, this%ndof) + 1
               tmp = tmp + real( this%coeff(j+this%half_stencil+1) * x_in(idx_w), 8 )
            end do
            x_out(idx) = real(tmp, c_float) + this%sigma * x_in(idx)
         end do

      end subroutine FDM1D_matvec

      ! Simple diagonal preconditioning
      subroutine FDM1D_precond(this, x_in, x_out)
         implicit none
         class(FDM1D_t), intent(inout) :: this
         real(c_float),   intent(in)   :: x_in(this%ndof)
         real(c_float),   intent(out)  :: x_out(this%ndof)
         integer(c_int32_t)            :: i
         real(c_float)                 :: diagcoeff

         diagcoeff = 1.0/( this%coeff( (this%ncoeff+1)/2 ) + this%sigma ) ! Diagonal coefficient is always the middle coefficient for a symmetric stencil, inverted

         !$acc parallel loop present(this%coeff) deviceptr(x_in, x_out)
         do i = 1, this%ndof
            x_out(i) = x_in(i) * diagcoeff
         end do
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
   integer(c_int32_t), parameter :: nNodes = 2001
   integer(c_int32_t), parameter :: maxIters = 1000
   integer(c_int32_t), parameter :: pOrder = 4
   real(c_double)    , parameter :: tol = 1.0e-7_c_double
   real(c_float)     , parameter :: pi = 3.14159265358979323846_c_float

   ! Internal vars
   integer(c_int32_t)         :: nWorking, i, j, k
   real(c_float)              :: err
   real(c_float), allocatable :: gridPts(:), x0(:), rhs(:), x_solve(:), Axsolve(:)
   type(FDM1D_t), target      :: laplObj
   type(c_ptr)                :: glassSolver
   type(c_ptr)                :: opData
   type(c_funptr)             :: matvecFunc
   type(c_funptr)             :: precondFunc

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

   ! Set object info
   laplObj%ndof         = nWorking
   laplObj%ncoeff       = pOrder + 1
   laplObj%half_stencil = pOrder / 2
   laplObj%dx           = (pi * 2.0_c_float) / real(nWorking, c_float)
   laplObj%sigma        = 1.0_c_float / (laplObj%dx**2)
   allocate(laplObj%coeff(laplObj%ncoeff))
   if (pOrder == 2) then
      laplObj%coeff = -[1.0_c_float, -2.0_c_float, 1.0_c_float] / (laplObj%dx**2)
   else if (pOrder == 4) then
      laplObj%coeff = -[-1.0_c_float/12.0_c_float, 4.0_c_float/3.0_c_float, -5.0_c_float/2.0_c_float, 4.0_c_float/3.0_c_float, -1.0_c_float/12.0_c_float] / (laplObj%dx**2)
   else
      print *, "Unsupported order"
      call MPI_Abort(client_comm, 1, ierr)
   end if

   ! Create grid points x0 and rhs
   ! x0 is going to be randomly initialized with values between -1 and 1, rhs is going to be sin(x) evaluated at the grid points
   allocate(gridPts(nWorking), rhs(nWorking), x0(nWorking))
   do i = 1, nWorking
      gridPts(i) = real( (i-1), c_float ) * laplObj%dx
      rhs(i) = (1.0_c_float+laplObj%sigma)*sin(gridPts(i))
      call random_number(x0(i))
      x0(i) = (2.0_c_float * x0(i)) - 1.0_c_float
   end do

   ! Create the GLASs solver
   glassSolver = cg_create_u32_pf(client_comm, nWorking, maxIters, tol)

   ! Setup x0 and b
   call cg_setup_u32_f(glassSolver, x0, rhs)

   ! Setup the matvec and preconditioner
   opData = c_loc(laplObj)
   matvecFunc = c_funloc(FDM1D_matvec_c)
   precondFunc = c_funloc(FDM1D_precond_c)

   ! Call the FPCG solver
   call fpcg_solve_u32_f(glassSolver, matvecFunc, precondFunc, opData)

   ! Get the solution back
   allocate(x_solve(nWorking))
   call cg_get_solution_u32_f(glassSolver, x_solve)

   ! Verification: ax - b < tol
   allocate(Axsolve(nWorking))
   call laplObj%matvec(x_solve, Axsolve)
   do i = 1, nWorking
      err = abs(Axsolve(i) - rhs(i))
      if (real(err,8) > tol*200.0_c_double) then
         print *, "Rank ", irank, ": Error at node ", i, " is ", err, " which exceeds tolerance ", tol*200.0_c_double
         call MPI_Abort(client_comm, 1, ierr)
      end if
   end do

   ! If passed, print the max error
   if (irank == 0) then
      print *, "Test passed. Maximum error is ", maxval(abs(Axsolve - rhs))
   end if

   ! Finalize MPI
   call MPI_Finalize(ierr)
end program test_32