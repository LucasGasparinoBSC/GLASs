module mod_Diffusion1D_Jacobian

    use mod_Diffusion1D_LGL
    use iso_c_binding, only: ip => c_int32_t, rp => c_float, dp => c_double

    type Diffusion1D_Jacobian_t
        integer(ip) p
        integer(ip) nelem
        real(rp) :: advectionVelocity
        real(rp) :: viscosity
        real(rp) :: domainSize

    contains
        procedure, public :: getLocalJacobian => mod_Diffusion1D_Jacobian_get_jacobian
		procedure, public :: getLocalImplicitOperator => mod_Diffusion1D_Jacobian_get_implicit_operator


    end type Diffusion1D_Jacobian_t
contains

    subroutine getBarycentricWeights(p, xvals, weights)
        ! calculates the Lagrange barycentric weights at each xval
		integer(ip), intent(in) :: p
        real(rp), intent(in) :: xvals(p+1)
        real(rp), intent(out) :: weights(p+1)
        integer(ip) :: j, k
        do j = 1, p + 1
            weights(j) = 1
        end do
        do j = 2, p + 1
            do k = 1, j-1
                weights(k) = weights(k)*(xvals(k) - xvals(j))
                weights(j) = weights(j)*(xvals(j) - xvals(k))
            end do
        end do
        do j = 1, p + 1
            weights(j) = 1/weights(j)
        end do
        return
    end subroutine

    subroutine getLocalDerivMatrix(p, nelem, derivMatrix, lagrangeWeights, refXVals)
        implicit none

        integer(ip), intent(in) :: p, nelem
		real(rp), intent(in) :: lagrangeWeights(p + 1), refXVals(p + 1)
        real(rp), intent(out) :: derivMatrix(p + 1, p + 1)

        integer(ip) :: i, j

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

    subroutine mod_Diffusion1D_Jacobian_get_jacobian(this, jacobian)
		implicit none
        class(Diffusion1D_Jacobian_t), intent(inout) :: this
        real(rp), intent(out) :: jacobian(this%p + 1, this%p + 1)

        real(rp) :: derivMat(this%p + 1, this%p + 1), diffuseMatrix(this%p + 1, this%p + 1)
        real(rp) :: lglWeights(this%p + 1), xVals(this%p + 1), lagrangeWeights(this%p + 1), invLglWeights(this%p+1)
        real(rp) :: deltaX
        integer(ip) :: j

        call LegendreGaussLobattoNodeEval(this%p, xVals, lglWeights)
        call getBarycentricWeights(this%p, xVals, lagrangeWeights)
        call getLocalDerivMatrix(this%p, this%nelem, derivMat, lagrangeWeights, xVals)


		deltaX = this%domainSize/this%nelem

		lglWeights = (deltaX/2) * lglWeights

		print *, "lagrangeWeights, deriv, deltaX"
		print *, xVals
		print *, lglWeights
		print *, lagrangeWeights
		print "(6F22.15)", derivMat
		print *, deltaX
		print *, "---"

        do j = 1, this%p + 1
			invLglWeights = 1./lglWeights(j)
			jacobian(j,:) = lglWeights(j)*derivMat(j,:)
        end do

		! account for nodes at element boundaries
		! their weight will be doubled later since they belong to two elements
		invLglWeights(1) = invLglWeights(1)/2
		invLglWeights(this%p+1) = invLglWeights(this%p+1)/2

        jacobian = -matmul(transpose(derivMat), jacobian)
		print "(6F22.15)", jacobian
		print *, "---"
		do j=1,this%p+1
		jacobian(j, :) = invLglWeights(j)*jacobian(j, :)
		end do

		print "(6F22.15)", transpose(jacobian)


        jacobian = this%viscosity*2./deltaX*jacobian

    end subroutine
	subroutine mod_Diffusion1D_Jacobian_get_implicit_operator(this, deltaT, implOp)
        class(Diffusion1D_Jacobian_t), intent(inout) :: this

		real(rp) :: implOp(this%p + 1, this%p + 1)
		real(rp) :: identity(this%p+1,this%p+1)
		real(rp) :: jacobian(this%p+1,this%p+1)


		integer i

		identity = 0 

		do i = 1, this%p + 1
			identity(i,i) = 1_rp
		end do

		! when applied globally, nodes at element boundaries will receive contributions from two elements
		! half the identity at these nodes to compensate
		identity(1,1) = 0.5_rp
		identity(this%p+1,this%p+1) = 0.5_rp 

		call this%getLocalJacobian(jacobian)
		implOp = identity - deltaT*jacobian
	end subroutine
end module
