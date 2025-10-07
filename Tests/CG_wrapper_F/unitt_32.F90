module cg_wrapper_mod
    use iso_c_binding
    implicit none
    interface
        ! Constructor call
        function cg_create_u32_f(arrSize, maxIters, tol) bind(C, name="cg_create_u32_f")
            import :: c_ptr, c_int32_t, c_double
            implicit none
            integer(c_int32_t), value :: arrSize
            integer(c_int32_t), value :: maxIters
            real(c_double), value :: tol
            type(c_ptr) :: cg_create_u32_f
        end function cg_create_u32_f

        ! Destructor call
        subroutine cg_destroy_u32_f(solver) bind(C, name="cg_destroy_u32_f")
            import :: c_ptr
            implicit none
            type(c_ptr), value :: solver
        end subroutine cg_destroy_u32_f

        ! Setup
        subroutine cg_setup_u32_f(solver, inicond, rhs) bind(C, name="cg_setup_u32_f")
            import :: c_ptr, c_float
            implicit none
            type(c_ptr), value :: solver
            real(c_float), intent(in) :: inicond(*)
            real(c_float), intent(in) :: rhs(*)
        end subroutine cg_setup_u32_f

        ! Solver call
        subroutine cg_solve_u32_f(solver, matvec, user_data) bind(C, name="cg_solve_u32_f")
            import :: c_ptr, c_funptr
            implicit none
            type(c_ptr), value :: solver
            type(c_funptr), value :: matvec
            type(c_ptr), value :: user_data
        end subroutine cg_solve_u32_f
    end interface
    ! Abstract interface for matvec
    abstract interface
        subroutine matvec_f32(x_in, x_out, user_data) bind(C)
            use iso_c_binding, only : c_ptr
            implicit none
            real(kind=4), intent(in) :: x_in(*)
            real(kind=4), intent(out) :: x_out(*)
            type(c_ptr), value :: user_data
        end subroutine matvec_f32
    end interface
end module cg_wrapper_mod

module fortranOps
    use iso_c_binding
    implicit none
    public :: diag_matvec_c
    contains
        ! Fortran concrete routine
        subroutine diag_matvec(x_in, x_out, diagA, n)
            implicit none
            integer(c_int32_t), value :: n
            real(c_float), intent(in) :: x_in(n)
            real(c_float), intent(in) :: diagA(n)
            real(c_float), intent(out) :: x_out(n)
            integer(4) :: i

            !$acc parallel loop deviceptr(x_in, x_out, diagA)
            do i = 1, n
                x_out(i) = diagA(i) * x_in(i)
            end do
            !$acc end parallel loop
        end subroutine diag_matvec

        ! C wrapper for the Fortran matvec
        subroutine diag_matvec_c(x_in, x_out, user_data) bind(C)
            use iso_c_binding, only : c_ptr, c_float
            implicit none
            real(c_float), intent(in) :: x_in(*)
            real(c_float), intent(out) :: x_out(*)
            type(c_ptr), value :: user_data

            type c_diag_data
                type(c_ptr) :: diagA_ptr
                integer(c_int32_t) :: n
            end type c_diag_data

            type(c_diag_data), pointer :: dMatrix
            real(c_float), pointer :: diagA(:)

            call c_f_pointer(user_data, dMatrix)
            call c_f_pointer(dMatrix%diagA_ptr, diagA, [dMatrix%n])

            call diag_matvec(x_in, x_out, diagA, dMatrix%n)
        end subroutine diag_matvec_c
end module fortranOps

program unitt_32
    use iso_c_binding
    use cg_wrapper_mod
    use fortranOps, only: diag_matvec_c
        implicit none

        ! Type definition for user_data
        type, bind(C) :: c_diag_data
            type(c_ptr) :: diagA_ptr
            integer(c_int32_t) :: n
        end type c_diag_data

        ! Variables
        integer(c_int32_t), parameter :: n = 1000000 ! Size of the system
        integer(c_int32_t), parameter :: maxIters = 10 ! Max iterations
        real(c_double), parameter :: tol = 1.0e-6 ! Tolerance
        type(c_ptr) :: cgSolver, user_data
        type(c_funptr) :: matvec_funptr
        real(c_float), allocatable, target :: x0(:), b(:), diagA(:)
        type(c_diag_data), target :: diag_info
        integer :: i

        ! Allocate and initialize vectors
        allocate(x0(n), b(n), diagA(n))
        do i = 1, n
            x0(i) = 0.001_c_float
            b(i) = 1.0_c_float
            diagA(i) = 4.0_c_float
        end do
        !$acc enter data copyin(diagA)

        ! Fill user_data
        !$acc host_data use_device(diagA)
        diag_info%diagA_ptr = c_loc(diagA)
        !$acc end host_data
        diag_info%n = n
        user_data = c_loc(diag_info)

        ! Create solver instance and setup
        cgSolver = cg_create_u32_f(n, maxIters, tol)
        call cg_setup_u32_f(cgSolver, x0, b)

        ! Get function pointer for matvec
        matvec_funptr = c_funloc(diag_matvec_c)

        ! Solve the system
        call cg_solve_u32_f(cgSolver, matvec_funptr, user_data)
end program unitt_32