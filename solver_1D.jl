using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")
Pkg.add("IterativeSolvers")
Pkg.add("Printf")
Pkg.add("CuArrays")
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Printf
using CuArrays
#CuArrays.allowscalar(false)
using CUDAnative
using CUDAdrv

Δz = 5e-5
z = 0:Δz:1
N = length(z)


function exact(z)
    return -1/2 * z.^2 + z
end

# create sparse matrix A
Il = 2:N-1
Jl = 1:N-2
Iu = 1:N-2
Ju = 2:N-1

dl = ones(N-2)
du = ones(N-2)
d = -2*ones(N-1)
A = sparse(Il,Jl,dl,N-1,N-1) +
    sparse(Iu,Ju,du,N-1,N-1) +
    sparse(1:N-1,1:N-1,d,N-1,N-1)
A[N-1, N-1] = -1
A .= A / (Δz^2)

b = -ones(N-1, 1)


#   Perform LU solve
println("Direct solve on CPU")
<<<<<<< HEAD:solver_1D.jl
Udummy = A \ b
@time Umod = A \ b
umod = [0;Umod]
@printf "norm (our solution - exact) = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(umod - exact(z))
=======
u_dummy_DSCPU = A \ b
@time u_int_DSCPU = A \ b
u_DSCPU = [0; u_int_DSCPU]
# @show umod
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_DSCPU - exact(z))
>>>>>>> e2e2afded96716304471b448a13dd6ad4cd1739f:solver.jl
println("-----------")
println()

#    Perform CG solve
println("Native Julia CG solve on CPU")
<<<<<<< HEAD:solver_1D.jl
u_dummy = cg(A, b)
@time u_1 = cg(A, b)
u = [0;u_1]
@printf "norm (our solution - exact) = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u - exact(z))
=======
u_dummy_CGCPU = cg(-A, -b)
@time u_int_CGCPU = cg(-A, -b)
u_CGCPU = [0; u_int_CGCPU]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_CGCPU - exact(z))
>>>>>>> e2e2afded96716304471b448a13dd6ad4cd1739f:solver.jl

println("-----------")
println()


d_A = CuArray(A)
d_b = CuArray(b)

println("Direct solve on GPU")
<<<<<<< HEAD:solver_1D.jl
dummy = d_A \ d_b
@time u_old = d_A \ d_b
regular_u = Array(u_old)
u = [0;regular_u]
@printf "norm (our solution - exact) = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u - exact(z))
=======
u_dummy_DSGPU = d_A \ d_b
@time u_int_DSGPU = d_A \ d_b
u_int_DSGPU_reg = Array(u_int_DSGPU)
u_DSGPU = [0; u_int_DSGPU_reg]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_DSGPU - exact(z))
>>>>>>> e2e2afded96716304471b448a13dd6ad4cd1739f:solver.jl
println("-----------")
println()

# CG solve on GPU
<<<<<<< HEAD:solver_1D.jl
d_A = CuArrays.CUSPARSE.CuSparseMatrixCSC(A)
d_b = CuArray(b)
u_cg = CuArray(zeros(size(d_b)))
u_dummy = CuArray(zeros(size(d_b)))

println("CG solve on GPU")
cg!(u_dummy, d_A, d_b)

@time cg!(u_cg, d_A, d_b)


regular = Array(u_cg)
u = [0;regular]
@printf "norm (our solution - exact) = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u - exact(z))
=======
#u_cg = CuArray(zeros(size(d_b)))
#u_dummy = CuArray(zeros(size(d_b)))

println("CG solve on GPU")
u_dummy_CGCPU = cg(-d_A, -d_b)
@time u_int_CGGPU = cg(-d_A, -d_b)
u_int_CGGPU_reg = Array(u_int_CGGPU)
u_CGGPU = [0; u_int_CGGPU_reg]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_CGGPU - exact(z))
>>>>>>> e2e2afded96716304471b448a13dd6ad4cd1739f:solver.jl
println("-----------")
println()
