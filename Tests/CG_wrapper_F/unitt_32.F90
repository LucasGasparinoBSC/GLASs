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

        ! Get solution
        function cg_get_solution_u32_f(solver) bind(C, name="cg_get_solution_u32_f")
            import :: c_ptr, c_float
            implicit none
            type(c_ptr), value :: solver
            real(c_float), pointer :: cg_get_solution_u32_f(:)
        end function cg_get_solution_u32_f

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
    end interface
end module cg_wrapper_mod

module cg_callbacks
    use iso_c_binding
    implicit none
    contains
        subroutine diag_matvec_acc(x_in, x_out, user_data, narr) bind(C)
            implicit none
            integer(c_int32_t), value :: narr
            real(c_float), intent(in) :: x_in(narr)
            real(c_float), intent(out) :: x_out(narr)
            real(c_float), pointer :: diag(:)
            type(c_ptr), value :: user_data
            integer(4) :: i, n

            n = size(x_in)
            call c_f_pointer(user_data, diag, [n])

            !$acc parallel loop
            do i = 1, n
                x_out(i) = diag(i) * x_in(i)
            end do
            !$acc end parallel loop
        end subroutine diag_matvec_acc
end module cg_callbacks

program unitt_32_ConjGrad
    use iso_c_binding
    use cg_wrapper_mod
    implicit none
    integer(c_int32_t), parameter :: nArrays = 10 ! Array size
    type(c_ptr) :: solver
    real(c_float), allocatable :: x0(:), b(:)

    ! Create a solver instance
    solver = cg_create_u32_f(nArrays, 100_c_int32_t, 1.0e-6_c_double)

    ! Setup inicond and rhs arrays (example values)
    allocate(x0(nArrays), b(nArrays))
    x0 = 0.001_c_float
    b = 1.0_c_float
    call cg_setup_u32_f(solver, x0, b)

    ! Dealloc fortran memory
    deallocate(x0, b)

    ! Destroy the solver instance
    call cg_destroy_u32_f(solver)
end program unitt_32_ConjGrad