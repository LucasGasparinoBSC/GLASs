module MATCSR_mod

   use iso_c_binding
   use cg_wrapper_mod

   implicit none

   type MATCSR_t
      integer(c_int32_t)          :: n = 0
      integer(c_int32_t)          :: nnz = 0
      integer(c_int32_t), pointer :: row(:) => null()
      integer(c_int32_t), pointer :: col(:) => null()
      real(c_float),      pointer :: val(:) => null()
   contains
      procedure, pass :: matvec => MATCSR_matvec
      procedure, pass :: halocomm => MATCSR_halocomm
      procedure, pass :: precond => MATCSR_precond
   end type

   public :: MATCSR_matvec_c, MATCSR_halocomm_c, MATCSR_precond_c

   contains

   ! Fortran concrete routine
   subroutine MATCSR_matvec(this, x_in, x_out)

      class(MATCSR_t), intent(inout) :: this
      real(c_float),   intent(in)    :: x_in(this%n)
      real(c_float),   intent(out)   :: x_out(this%n)
      integer(c_int32_t) :: i, j, k

      !$acc parallel loop present(this%row,this%col,this%val) deviceptr(x_in, x_out)
      do i = 1, this%n
         do k = this%row(i), this%row(i+1)-1
            j = this%col(k)
            x_out(i) = x_out(i) + this%val(k) * x_in(j)
         end do
      end do

   end subroutine MATCSR_matvec

   ! Fortran concrete halo communication routine (empty for testing)
   subroutine MATCSR_halocomm(this, x_inout)
      implicit none
      class(MATCSR_t), intent(inout) :: this
      real(c_float),   intent(inout) :: x_inout(this%n)
   end subroutine MATCSR_halocomm

   ! Fortran concrete preconditioner routine (diagonal for testing)
   subroutine MATCSR_precond(this, x_in, x_out)
      implicit none
      class(MATCSR_t), intent(inout) :: this
      real(c_float),   intent(in)    :: x_in(this%n)
      real(c_float),   intent(out)   :: x_out(this%n)
      integer(c_int32_t) :: i

      !$acc parallel loop present(this%row,this%col,this%val) deviceptr(x_in, x_out)
      do i = 1, this%n
         x_out(i) = x_in(i) / 2.0_c_float
         !x_out(i) = x_in(i) / this%val(this%row(i))
      end do
   end subroutine MATCSR_precond

   ! C wrapper for the Fortran matvec
   subroutine MATCSR_matvec_c(x_in, x_out, opData) bind(C)
      
      real(c_float), intent(in)  :: x_in(*)
      real(c_float), intent(out) :: x_out(*)
      type(c_ptr),   value       :: opData

      integer(c_int32_t) :: i
      type(MATCSR_t), pointer :: mat

      call c_f_pointer(opData, mat)

      !$acc parallel loop deviceptr(x_out)
      do i = 1, mat%n
         x_out(i) = 0.0_c_float
      end do

      call mat%matvec(x_in, x_out)

   end subroutine MATCSR_matvec_c

   ! C wrapper for the Fortran halo communication
   subroutine MATCSR_halocomm_c(x_inout, haloData) bind(C)
      implicit none
      real(c_float), intent(inout) :: x_inout(*)
      type(c_ptr),   value         :: haloData
      type(MATCSR_t), pointer :: mat

      call c_f_pointer(haloData, mat)
      call mat%halocomm(x_inout)
   end subroutine MATCSR_halocomm_c

   ! C wrapper for the Fortran preconditioner
   subroutine MATCSR_precond_c(x_in, x_out, precondData) bind(C)
      implicit none
      real(c_float), intent(in)  :: x_in(*)
      real(c_float), intent(out) :: x_out(*)
      type(c_ptr),   value       :: precondData
      type(MATCSR_t), pointer :: mat

      call c_f_pointer(precondData, mat)
      call mat%precond(x_in, x_out)
   end subroutine MATCSR_precond_c

end module MATCSR_mod


program unitt_32

   use mpi
   use iso_c_binding
   use cg_wrapper_mod
   use MATCSR_mod, only: MATCSR_matvec_c, MATCSR_halocomm_c, MATCSR_precond_c, MATCSR_t

   implicit none

   ! MPI variables
   integer :: ierr, crank, csize, client_comm

   integer(c_int32_t), parameter :: n = 1000          ! Size of the system
   integer(c_int32_t), parameter :: maxIters = 100000 ! Max iterations
   real(c_double),     parameter :: tol = 1.0e-7      ! Tolerance

   type(c_ptr)                :: cgSolver    ! GLASs CG solver
   type(c_ptr)                :: opData      ! Pointer to data to be passed to GLASs
   type(c_ptr)                :: haloData    ! Pointer to data to be passed to GLASs
   type(c_ptr)                :: precondData ! Pointer to data to be passed to GLASs
   type(c_funptr)             :: opFunction  ! Pointer to function to be passed to GLASs
   type(c_funptr)             :: haloFunction  ! Pointer to the halo comms function
   type(c_funptr)             :: precondFunction  ! Pointer to the preconditioner function
   real(c_float)              :: sref
   type(MATCSR_t), target     :: mat
   real(c_float), allocatable :: x0(:), b(:), s(:)

   integer(c_int32_t) :: i, k

   ! Initialize MPI
   call MPI_Init(ierr)

   ! Define a MPI communicator
   client_comm = MPI_COMM_WORLD
   call MPI_Comm_rank(client_comm, crank, ierr)
   call MPI_Comm_size(client_comm, csize, ierr)

   ! Create sparse CSR matrix
   mat%n = n
   mat%nnz = 3 * n - 2
   allocate(mat%val(mat%nnz), source=0.0_c_int32_t)
   allocate(mat%col(mat%nnz), source=0_c_int32_t)
   allocate(mat%row(n+1), source=0_c_int32_t)

   ! Fill sparse CSR matrix (tridiagonal)
   mat%row(1) = 1
   k = 1_c_int32_t
   do i = 1, n
      if( i == 1 )then
         mat%val(k) = 2.0_c_float
         mat%col(k) = 1
         k = k + 1
         mat%val(k) = -1.0_c_float
         mat%col(k) = 2
         k = k + 1
      elseif( i == n )then
         mat%val(k) = -1.0_c_float
         mat%col(k) = n - 1
         k = k + 1
         mat%val(k) = 2.0_c_float
         mat%col(k) = n
         k = k + 1
      else
         mat%val(k) = -1.0_c_float
         mat%col(k) = i - 1
         k = k + 1
         mat%val(k) = 2.0_c_float
         mat%col(k) = i
         k = k + 1
         mat%val(k) = -1.0_c_float
         mat%col(k) = i + 1
         k = k + 1
      endif
      mat%row(i+1) = k
   end do
   !$acc enter data copyin(mat%col, mat%row, mat%val)
   !$acc enter data copyin(mat)
   !$acc enter data attach(mat%col, mat%row, mat%val)

   ! Allocate and initialize vectors
   allocate(x0(n), source=0.0_c_float)
   allocate(b(n), source=1.0_c_float)
   allocate(s(n), source=0.0_c_float)
   !$acc enter data copyin(x0, b, s)

   ! Create solver instance and setup
   cgSolver = cg_create_u32_pf(client_comm, n, maxIters, tol)

   !$acc host_data use_device(x0, b)
   call cg_setup_u32_f(cgSolver, x0, b)
   !$acc end host_data

   ! Get function and data pointers
   opData = c_loc(mat)
   opFunction = c_funloc(MATCSR_matvec_c)
   precondFunction = c_funloc(MATCSR_precond_c)

   ! Solve the system
   !call cg_solve_u32_f(cgSolver, opFunction, opData)
   call fpcg_solve_u32_f(cgSolver, opFunction, precondFunction, opData)

   ! Recover the solution
   !$acc host_data use_device(s)
   call cg_get_solution_u32_f(cgSolver, s)
   !$acc end host_data
   !$acc update self(s)

   ! Testing the solution
   do i = 1, n
      sref = real(i*(n+1-i),kind=c_float)*0.5_c_float
      if( abs(s(i) - sref) / abs(sref) > 1.0e-7_c_float )then 
         write(*,*) "Test failed: solution is not correct:",i,s(i),sref,abs(s(i) - sref)/abs(sref)
         stop 1
      endif
   enddo

   write(*,*) "Test passed: solution is approximately correct."
   !stop 0;

   ! Destroy solver instance
   call cg_destroy_u32_f(cgSolver)

   ! Deallocate arrays
   !$acc exit data delete(mat%col, mat%row, mat%val)
   !$acc exit data delete(mat)
   !$acc exit data delete(x0, b, s)
   deallocate(mat%val, mat%col, mat%row)
   deallocate(x0, b, s)

   ! Finalize MPI
   call MPI_Finalize(ierr)

end program unitt_32