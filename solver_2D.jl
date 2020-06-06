#=
using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")
Pkg.add("IterativeSolvers")
Pkg.add("Printf")
Pkg.add("CuArrays")
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")
=#

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Printf

using CuArrays
CuArrays.allowscalar(false)
using CUDAnative
using CUDAdrv


Δz = 0.1
z = 0:Δz:1
n = length(z)
N = (n-1)^2


# create sparse matrix A
Il = 2:N
Jl = 1:N-1
Iu = 1:N-1
Ju = 2:N

dl = ones(N-1)
du = ones(N-1)
d = -4 * ones(N)
A = sparse(Il,Jl,dl,N,N) + sparse(Iu,Ju,du,N,N) + sparse(1:N,1:N,d,N,N)

# rows α(N-1) the diagonal entry is -3
for α=1:n-2
    A[α*(n-1),α*(n-1)] = -3
end

# rows (N-2)(N-1)+β the diagonal is -3
for β=1:n-2
    A[(n-2)*(n-1)+β,(n-2)*(n-1)+β] = -3
end

# rows (N-1)^2 the diagonal is -2
A[(n-1)^2,(n-1)^2] = -2

# locations (α*(n-1)+1, α*(n-1)) should be zero
for α=1:n-2
    A[α*(n-1)+1,α*(n-1)] = 0
end

# locations (α*(n-1), α*(n-1)+1) should be zero
for α=1:n-2
    A[α*(n-1),α*(n-1)+1] = 0
end

# print(Matrix(A))
# print(size(A))

b = -Δz * ones(N, 1)
# print(b)
# print(size(b))

#   Perform LU solve
println("Direct solve on CPU")
u_dummy_DSCPU = A \ b
@time u_int_DSCPU = A \ b
u_DSCPU = reshape(u_int_DSCPU, (n-1, n-1))
u_DSCPU = transpose(u_DSCPU)
print(size(u_DSCPU))
# @show umod
# @printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" norm(u_DSCPU - exact(z))
println("-----------")
println()


#    Perform CG solve
println("Native Julia CG solve on CPU")
u_dummy_CGCPU = cg(-A, -b)
@time u_int_CGCPU = cg(-A, -b)
u_DSCPU = reshape(u_int_DSCPU, (n-1, n-1))
u_DSCPU = transpose(u_DSCPU)
# @printf "norm between our solution and the exact solution = \x1b[31m %e \x1b[0m\n" sqrt(Δz) * norm(u_CGCPU - exact(z))

println("-----------")
println()
