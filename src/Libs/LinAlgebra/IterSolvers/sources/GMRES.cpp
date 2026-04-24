#include "GMRES.hpp"

template <typename ITYPE, typename RTYPE>
// empty constructor for flexibility in initialization
GMRES<ITYPE, RTYPE>::GMRES() : IterSolvers<ITYPE, RTYPE>() {
    PUSH_RANGE("GMRES::Constructor(empty)", 4);
    this->restart = 0;
    this->g = nullptr;
    this->cs = nullptr;
    this->sn = nullptr;
    this->qj_tmp = nullptr;

    POP_RANGE();
}

// constructor with parameters and communicator
template <typename ITYPE, typename RTYPE>
GMRES<ITYPE, RTYPE>::GMRES(MPI_Comm& c_comm, ITYPE arrSize, ITYPE maxIters, double tol, ITYPE restart) 
: IterSolvers<ITYPE, RTYPE>(c_comm, arrSize, maxIters, tol),
  arnoldi(c_comm, arrSize, restart) // Initialize Arnoldi iteration object with same communicator and restart size
  
  {
    if (this->IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: using GMRES solver!" << std::endl;
    PUSH_RANGE("GMRES::Constructor(param+comm)", 4);

    this->restart = restart;
    // Allocate host memory using calloc (ensures init to 0)
    this->g  = (double*)calloc(restart + 1, sizeof(double));
    this->cs = (double*)calloc(restart,     sizeof(double));
    this->sn = (double*)calloc(restart,     sizeof(double));
    this->qj_tmp = (RTYPE*)calloc(this->arrSize, sizeof(RTYPE));

    // Start logfile
    this->logfile_name = "GLASs_gmresSolver";
    if (this->IterSolvers_comm.getLibRank() == 0) {
        // Open the file for writing
        this->logfile.open(this->logfile_name + this->logfile_ext, std::ios::out);
        // Wrtie the header: "ITER ||rk|| gmresTime(ms)"
        this->logfile << "ITER --- "
                      << " --- ||rk|| --- "
                      << " --- gmresTime(ms)" << std::endl;
        this->logfile << "-------------------------------" << std::endl;
        this->logfile.flush();
    }
    POP_RANGE();
}

// Desctructor
template <typename ITYPE, typename RTYPE>
GMRES<ITYPE, RTYPE>::~GMRES() {
    if (this->IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: destroying GMRES solver" << std::endl;

    // Close logfile
    if (this->IterSolvers_comm.getLibRank() == 0) this->logfile.close();

    PUSH_RANGE("GMRES::Destructor", 4);
    // Free host memory
    if (this->g)  free(this->g);
    if (this->cs) free(this->cs);
    if (this->sn) free(this->sn);
    if (this->qj_tmp) free(this->qj_tmp);

    #ifdef USE_GPU
        // Free device memory
        DeviceUtils::free_device(this->d_g);
        DeviceUtils::free_device(this->d_cs);
        DeviceUtils::free_device(this->d_sn);
        DeviceUtils::free_device(this->d_qj_tmp);
    #endif
    POP_RANGE();
}

//--------------------------//
// Helper Functions for GMRES solver //
//--------------------------//

// applyGivens - applies Givens rotations to column j of H and updates g
// H from tridiagonal is converted to upper triangular, and g is updated with the same rotations
template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::applyGivens(int j) {
    
    // Get pointers from H to ArnoldIter
    double* H = this->arnoldi.getUpHess();
    ITYPE m = this->restart;

    // 1. Apply previous Givens rotations to the new column of H
    for (ITYPE i = 0; i < j; i++) {
        double tmp = this->cs[i]*(double)H[i*m + j] 
                    + this->sn[i]*(double)H[(i+1)*m + j];
        H[(i+1)*m + j] = (-this->sn[i]*(double)H[i*m + j] 
                                + this->cs[i]*(double)H[(i+1)*m + j]);
        H[i*m + j] = tmp;
    }

    // 2. Compute new Givens rotation for element j
    double r = std::sqrt((double)H[j*m + j]*(double)H[j*m + j] + 
                        (double)H[(j+1)*m + j]*(double)H[(j+1)*m + j]);
    this->cs[j] = (double)H[j*m + j] / r;
    this->sn[j] = (double)H[(j+1)*m + j] / r;

    // 3. Apply new Givens rotation to column j of H
    H[j*m + j] = r;
    H[(j+1)*m + j] = 0.0;

    // 4. Apply new Givens rotation to g
    double tmp = this->cs[j]*this->g[j];
    this->g[j+1] = -this->sn[j]*this->g[j];
    this->g[j] = tmp;
}


// Solves the small upper triangular system H*y = g for y, where H is the upper Hessenberg matrix from Arnoldi iteration, 
// and g is the updated right-hand side after Givens rotations
template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::backSubstitution(int m, double *y) {

    // Get pointer to H from ArnoldIter
    double* H = this->arnoldi.getUpHess();
    ITYPE col = this->restart;

    // Solve from bottom to top
    for (int i = m; i >= 0; i--) {
        y[i] = this->g[i];
        for (int k = i+1; k <= m; k++)
            y[i] -= H[i*col + k]*y[k];
        y[i] /= H[i*col + i];
    }
}

// updateSolution - updates the solution x_sol = x_sol + Q*y, where Q is the Krylov basis from Arnoldi iteration,
//  and y is the solution of H*y=g
template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::updateSolution(int m, double *y) {

    // Get pointer to Q from ArnoldIter
    RTYPE* Q = this->arnoldi.getQkrylov();
    ITYPE col = this->restart + 1;

    // x = x + Q*y
    for (ITYPE i = 0; i < this->arrSize; i++) {
        double sum = 0.0;
        
        for (ITYPE j = 0; j <= m; j++)
            sum += (double)Q[i*col + j] * y[j];  // accumulate in double

        //this->x_sol[i] += (RTYPE)sum;  // convert once at the end
        this->x_sol[i] = (RTYPE)((double)this->x_sol[i] + sum);  // add in double, then cast
    }
}

//--------------------------//
// Main GMRES solver function //
//--------------------------//  

template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::gmresSolver(const MatVecOp& matvec, const HaloOp& halocomm) 
{
    double out_sqrtRes = static_cast<double>(0);
    double gmresTime = this->IterSolvers_comm.timeFunction([&]
    {
        PUSH_RANGE("GMRES::gmresSolver", 4);
        // For now, haloOp is not used in GMRES, but it can be passed to the matvec operator if needed for halo exchanges
        this->iter = 0;

        #ifdef USE_GPU
            // GPU implementation - to be developed later
        #else
            // CPU MPI implementation

            // Outer loop over restarts
            while (this->iter < this->maxIters)
            {
                // 1. x_sol = x0
                TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->x0, this->x_sol); // x_sol = x0

                // 2. Apply halo exchange to x_sol
                halocomm(this->x_sol); // Halo exchange for x_sol if needed

                // 3. Ax = A*x_sol
                matvec(this->x_sol, this->Ax); 

                // 4. r = b - Ax
                for (ITYPE i = 0; i < this->arrSize; i++)
                    this->rk[i] = this->b[i] - this->Ax[i]; 

                // 5. beta = |r| and setup q[0] = r/beta
                // ArnoldiIter.setup() computes beta and initializes q[0]
                // NOTE if double with RTYPE, it should be static cast to RTYPE
                this->arnoldi.setup(this->rk); 

                // 6. Compute beta = //r// and check convergence
                // recompute norm of rk for res0
                double local = 0.0;
                for (ITYPE i = 0; i < this->arrSize; i++)
                    local += (double)this->rk[i] * (double)this->rk[i];
                MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_DOUBLE, MPI_SUM, this->IterSolvers_comm.getLibComm());
                double beta = std::sqrt(local);

                // 7. Check convergence
                double res0 = this->tol * beta;
                if (beta <= res0){
                    out_sqrtRes = beta;
                    break; // Converged, exit restart loop  
                }

                // 8. Initialize g = beta*e1
                for (ITYPE i = 0; i < this->restart+1; i++)
                    this->g[i] = 0.0;
                this->g[0] = beta;

                // 9. Reset cs and sn 
                for (ITYPE i = 0; i < this->restart; i++){
                    this->cs[i] = this->sn[i] = 0.0;    
                }

                int j = 0;
                // 10. Inner loop over Arnoldi iterations
                while (j < this->restart && this->iter < this->maxIters)
                {   
                    this->iter++;

                    // 10.1 Apply halo exchange to q[j], q[j] lives inside arnoldi.Qkrylov
                    // extract it, apply halo exchange, and put it back
                    RTYPE* Q = this->arnoldi.getQkrylov();
                    ITYPE col = this->restart + 1;

                    // temporary vector for halo exchange
                    for (ITYPE i = 0; i < this->arrSize; i++)                        
                        this->qj_tmp[i] = Q[i*col + j]; // copy q[j] to temporary vector

                    halocomm(this->qj_tmp); // halo exchange for q[j] if needed
                    for (ITYPE i = 0; i < this->arrSize; i++)
                        Q[i*col + j] = this->qj_tmp[i]; // put back q[j]

                    // 10.2 Arnoldi step at iteration j
                    // computes w = A*q[j], orthogonalizes, stores q[j+1] and returns h[j+1][j]
                    double h_new = this->arnoldi.arnoldiStep(j, matvec);

                    // 10.3 Apply Givens rotations to the new column of H and update g
                    this->applyGivens(j);

                    // 10.4 Current resiidual norm is |g[j+1]|, check convergence
                    out_sqrtRes = std::abs(this->g[j+1]);
                    if (out_sqrtRes <= res0){
                        j++;
                        break; // Converged
                    }
                    j++;
                }
                // END INNER LOOP

                // 11. Solve upper triangular system H*y = g for y
                double *y = (double *)calloc(j+1, sizeof(double));
                backSubstitution(j-1, y);
                
                // 12. Update solution x_sol = x_sol + Q*y
                updateSolution(j-1, y);

                // Free y   
                free(y);

                // Check convergence after solution update
                if (out_sqrtRes <= res0){
                    break; // Converged, exit restart loop  
                }
            }
            // END OUTER LOOP
        #endif
        POP_RANGE(); // 4
    });
    
    // Log final iteration, residual norm, and time, same pattern as CG
    if (this->IterSolvers_comm.getLibRank() == 0) {
        this->logfile << this->iter << " , " 
                      << std::scientific << out_sqrtRes 
                      << " , " << gmresTime*1000 << std::endl;
        this->logfile.flush();
    }
}


template class GMRES<uint32_t, float>;
template class GMRES<uint64_t, float>;
template class GMRES<uint32_t, double>;
template class GMRES<uint64_t, double>;
#ifdef USE_GPU
    template class GMRES<uint32_t, DeviceUtils::bf16>;
    template class GMRES<uint64_t, DeviceUtils::bf16>;
#endif
