using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Printf
using CuArrays
using CUDAnative
using CUDAdrv: synchronize

N = 100
Δt = 0.1

"""
    gpugesv!(A,b)

LU solve on GPU

"""
function gpugesv!(A,b)
    A, ipiv = CuArrays.CUSOLVER.getrf!(A)
    return CuArrays.CUSOLVER.getrs!('N',A,ipiv,b)
end

#=
    create sparse matrix A
=#
Il = 2:N
Jl = 1:N-1
Iu = 1:N-1
Ju = 2:N

dl = -ones(N-1)
du = -ones(N-1)
d = 2*ones(N)
A = sparse(Il,Jl,dl,N,N) + sparse(Iu,Ju,du,N,N) + sparse(1:N,1:N,d,N,N)
d_A = CuArray(A/Δt)

b = -ones(N, 1)
d_b = CuArray(b)


u = gpugesv!(d_A, d_b)
print(u)

# CG solve
A = cu(A)
A = A + A' + 2*N*I
b = cu(-ones(N))
x = cg(A, b)
print(x)
