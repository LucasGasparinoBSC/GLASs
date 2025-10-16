module mod_AdvectionDiffusion1D_Jacobian

    use type_inputParams
    use mod_lagrangebaryweights
    use mod_AdvectionDiffusion1D_LGL
    use iso_c_binding

    type mod_AdvectionDiffusion1D_Jacobian_t
        integer(c_int32_t) p
        integer(c_int32_t) nelem
        real(c_float) :: advectionVelocity
        real(c_float) :: viscosity
        real(c_float) :: domainSize

    contains
        procedure, public :: getLocalJacobian

    end type mod_AdvectionDiffusion1D_Jacobian_t
contains

    subroutine getBarycentricWeights(N, xvals, weights)
        ! calculates the Lagrange barycentric weights at each xval
        real(c_float), intent(in) :: xvals(N)
        real(c_float), dimension(:) :: weights(N)
        integer(c_int32_t) :: N
        integer(c_int32_t) :: j, k
        do j = 1, N + 1
            weights(j) = 1
        end do
        do j = 2, N + 1
            do k = 1, j
                weights(k) = weights(k)*(xvals(k) - xvals(j))
                weights(j) = weights(j)*(xvals(j) - xvals(k))
            end do
        end do
        do j = 1, N + 1
            weights(j) = 1/weights(j)
        end do
        return
    end subroutine

    subroutine getLocalDerivMatrix(p, nelem, derivMatrix, lagrangeWeights, refXVals)
        implicit none

        integer(c_int32_t) :: p, nelem
        real(c_float) :: derivMatrix(p + 1, p + 1)
        real(c_float), dimension(:) :: lagrangeWeights(p + 1), refXVals(p + 1)

        integer(c_int32_t) :: i, j

        derivMatrix = 0
        do i = 1, p + 1
            derivMatrix(i, i) = 0
            do j = 1, p + 1
                if (i /= j) then
                    derivMatrix(i, j) = lagrangeWeights(j)/( &
                                        lagrangeWeights(i)*(refXVals(i) - refXVals(j)) &
                                        )
                    derivMatrix(i, i) = derivMatrix(i, i) - derivMatrix(i, j)

                end if
            end do
        end do
        return
    end subroutine getLocalDerivMatrix

    subroutine getLocalJacobian(this, jacobian)
        class(mod_AdvectionDiffusion1D_Jacobian_t) :: this
        real(c_float) :: advMatrix(this%p + 1, this%p + 1), diffuseMatrix(this%p + 1, this%p + 1)
        real(c_float) :: jacobian(this%p + 1, this%p + 1)
        real(c_float) :: lobattoWeights(this%p + 1), xVals(this%p + 1), lagrangeWeights(this%p + 1)
        real(c_float) :: deltaX
        integer(c_int32_t) :: j

        call LegendreGaussLobattoNodeEval(N, xVals, lobattoWeights)
        call getBarycentricWeights(N, xVals, lagrangeWeights)
        call getLocalDerivMatrix(this%p, this%nelem, advMatrix, lagrangeWeights, xVals)

        do j = 1, this%p*this%nelem + 1
            jacobian(j, :) = lobattoWeights(j)*advMatrix(j, :)
        end do
        diffuseMatrix = matmul(-transpose(advMatrix), jacobian)
        advMatrix = jacobian
        deltaX = this%domainSize/this%nelem

        advMatrix = this%advectionVelocity*2./deltaX*advMatrix
        diffuseMatrix = this%viscosity*2./deltaX*diffuseMatrix

    end subroutine
end module
