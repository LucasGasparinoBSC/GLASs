module fortranOps

    use iso_c_binding
    use cg_wrapper_mod

    implicit none

    type MATDiag_t
        integer(c_int32_t) :: n
        real(c_float), pointer :: vA(:)
    contains
        procedure, pass :: matvec => MATDiag_matvec
    end type

    public :: MATDiag_matvec_c

    contains

    ! Fortran concrete routine
    subroutine MATDiag_matvec(this, x_in, x_out)
        implicit none
        class(MATDiag_t), intent(inout) :: this
        real(c_float),    intent(in)    :: x_in(this%n)
        real(c_float),    intent(out)   :: x_out(this%n)
        integer(4) :: i

        !$acc parallel loop deviceptr(x_in, x_out) present(this%vA)
        do i = 1, this%n
            x_out(i) = this%vA(i) * x_in(i)
        end do

    end subroutine MATDiag_matvec

    ! C wrapper for the Fortran matvec
    subroutine MATDiag_matvec_c(x_in, x_out, user_data) bind(C)
        use iso_c_binding, only : c_ptr, c_float
        implicit none
        real(c_float), intent(in)  :: x_in(*)
        real(c_float), intent(out) :: x_out(*)
        type(c_ptr),   value       :: user_data

        type(MATDiag_t), pointer :: dMatrix

        call c_f_pointer(user_data, dMatrix)
        call dMatrix%matvec(x_in, x_out)

    end subroutine MATDiag_matvec_c

end module fortranOps

program unitt_32

    use iso_c_binding
    use cg_wrapper_mod
    use fortranOps, only: MATDiag_matvec_c, MATDiag_t

    implicit none

    integer(c_int32_t), parameter :: n = 1000000     ! Size of the system
    integer(c_int32_t), parameter :: maxIters = 10   ! Max iterations
    real(c_double),     parameter :: tol = 1.0e-6    ! Tolerance

    type(c_ptr) :: cgSolver, user_data
    type(c_funptr) :: matvec_funptr
    real(c_float), allocatable :: x0(:), b(:), s(:)
    type(MATDiag_t), target :: mat
    integer :: i

    ! Fill user_data
    mat%n = n
    allocate(mat%vA(n), source=4.0_c_float)
    !$acc enter data copyin(mat, mat%vA)
    user_data = c_loc(mat)

    ! Allocate and initialize vectors
    allocate(x0(n), source=0.001_c_float)
    allocate(b(n), source=1.0_c_float)
    allocate(s(n), source=0.0_c_float)
    !$acc enter data copyin(x0, b, s)

    print*,'Metrics in the host before:',norm2(s)

    ! Create solver instance and setup
    cgSolver = cg_create_u32_f(n, maxIters, tol)
    !$acc host_data use_device(x0, b)
    call cg_setup_u32_f(cgSolver, x0, b)
    !$acc end host_data

    ! Get function pointer for matvec
    matvec_funptr = c_funloc(MATDiag_matvec_c)

    ! Solve the system
    call cg_solve_u32_f(cgSolver, matvec_funptr, user_data)

    ! Recover the solution
    !$acc host_data use_device(s)
    call cg_get_solution_u32_f(cgSolver, s)
    !$acc end host_data
    !$acc update self(s)

    print*,'Metrics in the host after:',norm2(s)

    ! Testing the solution
    do i = 1, n
        if( abs(s(i) - b(i)/mat%vA(i)) > 1.0e-6_c_float )then
            write(*,*) "Test failed: solution is not correct:",i,s(i),b(i)/mat%vA(i)
            stop 1
        endif
    enddo

    write(*,*) "Test passed: solution is approximately correct."

end program unitt_32