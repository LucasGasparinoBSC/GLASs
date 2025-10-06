module cg_wrapper_mod
    use iso_c_binding
    implicit none
    interface
        function cg_create_u32_f(arrSize, maxIters, tol) bind(C, name="cg_create_u32_f")
            import :: c_ptr, c_int32_t, c_double
            implicit none
            integer(c_int32_t), value :: arrSize
            integer(c_int32_t), value :: maxIters
            real(c_double), value :: tol
            type(c_ptr) :: cg_create_u32_f
        end function cg_create_u32_f
    end interface
end module cg_wrapper_mod

program unitt_32_ConjGrad
    use iso_c_binding
    use cg_wrapper_mod
    implicit none
    type(c_ptr) :: solver

    solver = cg_create_u32_f(1000_c_int32_t, 100_c_int32_t, 1.0e-6_c_double)
end program unitt_32_ConjGrad