using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Printf
# using CUDA
using CuArrays
CuArrays.allowscalar(false)
using CUDAnative
using CUDAdrv: synchronize

Δz = 0.0001
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
Udummy = A \ b
@time Umod = A \ b
umod = [0;Umod]
# @show umod
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(umod - exact(z))
println("-----------")
println()

#    Perform CG solve
println("Native Julia CG solve on CPU")
u_dummy = cg(A, b)
@time u_1 = cg(A, b)
u = [0;u_1]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u - exact(z))

println("-----------")
println()


d_A = CuArray(A)
d_b = CuArray(b)

println("Direct solve on GPU")
dummy = d_A \ d_b
@time u_old = d_A \ d_b
regular_u = Array(u_old)
u = [0;regular_u]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u - exact(z))
println("-----------")
println()

# CG solve on GPU
d_A = CuArray(A)
d_b = CuArray(b)
u_cg = CuArray(zeros(size(d_b)))
u_dummy = CuArray(zeros(size(d_b)))

println("CG solve on GPU")
cg!(u_dummy, d_A, d_b)
@time cg!(u_cg, d_A, d_b)

regular = Array(u_cg)
u = [0;regular]
@printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u - exact(z))
println("-----------")
println()
