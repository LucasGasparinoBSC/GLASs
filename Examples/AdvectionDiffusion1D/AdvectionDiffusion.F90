program advectionDiffusion
    use mod_AdvectionDiffusion1D_Jacobian
    use mod_AdvectionDiffusion1D_Base

    integer(c_int32_t) :: p, nelem, npoin
    real(c_float) :: advectionVelocity, viscosity
    class(AdvectionDiffusion1D_Base_t)::advDiff

    advDiff%AdvectionDiffusion1D_solve()
end program
