using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")
Pkg.add("IterativeSolvers")
Pkg.add("Printf")
Pkg.add("CuArrays")
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")
Pkg.add("Plots")

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Printf

using CuArrays
CuArrays.allowscalar(false)
using CUDAnative
using CUDAdrv
using Plots

# this shows the diff between CG on CPU and GPU
# Δz = 5e-5

Δz = 0.0125
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
A = sparse(Il,Jl,dl,N-1,N-1) + sparse(Iu,Ju,du,N-1,N-1) + sparse(1:N-1,1:N-1,d,N-1,N-1)
A[N-1, N-1] = -1
A .= A / (Δz^2)

b = -ones(N-1, 1)


#   Perform LU solve
println("Direct solve on CPU")
u_dummy_DSCPU = A \ b
@time u_int_DSCPU = A \ b
u_DSCPU = [0; u_int_DSCPU]
# @show umod
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_DSCPU - exact(z))
println("-----------")
println()

#    Perform CG solve
println("Native Julia CG solve on CPU")
u_dummy_CGCPU = cg(-A, -b)
@time u_int_CGCPU = cg(-A, -b)
u_CGCPU = [0; u_int_CGCPU]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_CGCPU - exact(z))

println("-----------")
println()


# d_A = CuArrays.CUSPARSE.CuSparseMatrixCSC(A)
d_A = CuArray(A)
d_b = CuArray(b)

println("Direct solve on GPU")
u_dummy_DSGPU = d_A \ d_b
@time u_int_DSGPU = d_A \ d_b
u_int_DSGPU_reg = Array(u_int_DSGPU)
u_DSGPU = [0; u_int_DSGPU_reg]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_DSGPU - exact(z))
println("-----------")
println()

# CG solve on GPU
d_A = CuArrays.CUSPARSE.CuSparseMatrixCSC(A)
d_b = CuArray(b)
u_cg = CuArray(zeros(size(d_b)))
u_dummy = CuArray(zeros(size(d_b)))

println("CG solve on GPU")
# u_dummy_CGCPU = cg(-d_A, -d_b)
cg!(u_dummy, d_A, d_b)
# @time u_int_CGGPU = cg(-d_A, -d_b)
@time cg!(u_cg, d_A, d_b)
u_int_CGGPU_reg = Array(u_cg)
u_CGGPU = [0; u_int_CGGPU_reg]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_CGGPU - exact(z))
println("-----------")
println()


# plot u against z
# u (x axis) z (y axis)
# plot(u_DSCPU,z)
# png("1D_mid_res")
