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
#ifdef USE_GPU
            real(c_float), device, intent(in) :: inicond(*)
            real(c_float), device, intent(in) :: rhs(*)
#else
            real(c_float), intent(in) :: inicond(*)
            real(c_float), intent(in) :: rhs(*)
#endif
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
        subroutine cg_get_solution_u32_f(solver, sol) bind(C, name="cg_get_solution_u32_f")
            import :: c_ptr, c_float
            implicit none
            type(c_ptr), value :: solver
#ifdef USE_GPU
            real(c_float), device, intent(out) :: sol(*)
#else
            real(c_float), intent(out) :: sol(*)
#endif
        end subroutine cg_get_solution_u32_f
        
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