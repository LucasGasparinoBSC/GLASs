module mod_AdvectionDiffusion1D_Jacobian

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
        real(c_float) :: derivMat(this%p + 1, this%p + 1), diffuseMatrix(this%p + 1, this%p + 1)
        real(c_float) :: jacobian(this%p + 1, this%p + 1)
        real(c_float) :: lglWeights(this%p + 1), xVals(this%p + 1), lagrangeWeights(this%p + 1), invLglWeights(this%p+1)
        real(c_float) :: deltaX
        integer(c_int32_t) :: j

        call LegendreGaussLobattoNodeEval(N, xVals, lglWeights)
        call getBarycentricWeights(N, xVals, lagrangeWeights)
        call getLocalDerivMatrix(this%p, this%nelem, derivMat, lagrangeWeights, xVals)

		deltaX = this%domainSize/this%nelem

		lglWeights = (deltaX/2) * lglWeights

        do j = 1, this%p + 1
			invLglWeights = 1./lglWeights(j)
        end do

		! account for nodes at element boundaries
		! their weight will be doubled later since they belong to two elements
		invLglWeights(1) = invLglWeights(1)/2
		invLglWeights(this%p+1) = invLglWeights(this%p+1)/2

        jacobian = matmul(transpose(derivMat), jacobian)

		do j=1,this%p+1
		jacobian(j, :) = invLglWeights(j)*jacobian(j, :)
		end do

        jacobian = this%viscosity*2./deltaX*jacobian

    end subroutine
end module
