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

Δz = 0.01
z = 0:Δz:1
y = 0:Δz:1
N = length(z)

# create sparse matrix A
Imain = 1:(N-1)^2
Jmain = 1:(N-1)^2

Il = 2:(N-1)^2
Jl = 1:((N-1)^2-1)
Iu = 1:((N-1)^2-1)
Ju = 2:(N-1)^2

il = N:(N-1)^2
jl = 1:((N-1)^2-(N-1))
iu = 1:((N-1)^2-(N-1))
ju = N:(N-1)^2

dbig = ones((N-1)^2-1)
dsmall = ones((N-1)^2-(N-1))
dmain = -4*ones((N-1)^2)
A = (sparse(Il,Jl,dbig,(N-1)^2,(N-1)^2)
    + sparse(Iu,Ju,dbig,(N-1)^2,(N-1)^2)
    + sparse(Imain,Jmain,dmain,(N-1)^2,(N-1)^2)
    + sparse(il,jl,dsmall,(N-1)^2,(N-1)^2)
    + sparse(iu,ju,dsmall,(N-1)^2,(N-1)^2))

# rows α(N-1) the diagonal entry is -3
for α=1:N-2
    A[α*(N-1),α*(N-1)] = -3
end

# rows (N-2)(N-1)+β the diagonal is -3
for β=1:N-2
    A[(N-2)*(N-1)+β,(N-2)*(N-1)+β] = -3
end

# rows (N-1)^2 the diagonal is -2
A[(N-1)^2,(N-1)^2] = -2

# locations (α*(n-1)+1, α*(n-1)) should be zero
# locations (α*(n-1), α*(n-1)+1) should be zero
for α=1:N-2
    A[α*(N-1)+1,α*(N-1)] = 0
    A[α*(N-1),α*(N-1)+1] = 0
end

# print(Matrix(A))
# print(size(A))

b = -Δz^2 * ones((N-1)^2, 1)
# print(b)
# print(size(b))

#   Perform LU solve
println("Direct solve on CPU")
u_dummy_DSCPU = A \ b
@time u_int_DSCPU = A \ b
u_DSCPU = reshape(u_int_DSCPU, (N-1, N-1))
U_DSCPU = zeros(N,N)
U_DSCPU[2:N,2:N] = u_DSCPU[:,:]
#u_DSCPU = transpose(u_DSCPU)
# print(size(u_DSCPU))
# @show umod
@printf "norm between AU and b = \x1b[31m %e \x1b[0m\n" norm(A*u_int_DSCPU - b)
println("-----------")
println()


#    Perform CG solve
println("Native Julia CG solve on CPU")
u_dummy_CGCPU = cg(-A, -b)
@time u_int_CGCPU = cg(-A, -b)
u_CGCPU = reshape(u_int_CGCPU, (N-1, N-1))
U_CGCPU = zeros(N,N)
U_CGCPU[2:N,2:N] = u_CGCPU[:,:]
#u_CGCPU = transpose(u_CGCPU)
@printf "norm between AU and b = \x1b[31m %e \x1b[0m\n" norm(A*u_int_CGCPU - b)

println("-----------")
println()

#=
# d_A = CuArrays.CUSPARSE.CuSparseMatrixCSC(A)
d_A = CuArray(A)
d_b = CuArray(b)

println("Direct solve on GPU")
u_dummy_DSGPU = d_A \ d_b
@time u_int_DSGPU = d_A \ d_b
u_DSGPU = Matrix(reshape(u_int_DSGPU, (N-1, N-1)))
U_DSGPU = zeros(N,N)
U_DSGPU[2:N,2:N] = u_DSGPU[:,:]
@printf "norm between AU and b = \x1b[31m %e \x1b[0m\n" norm(d_A*u_int_DSGPU - d_b)
println("-----------")
println()
=#



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
u_CGGPU = Matrix(reshape(u_cg, (N-1, N-1)))
U_CGGPU = zeros(N,N)
U_CGGPU[2:N,2:N] = u_CGGPU[:,:]
@printf "norm between AU and b = \x1b[31m %e \x1b[0m\n" norm(d_A*u_cg - d_b)
println("-----------")
println()


# u for direct solve on CPU (surface)

# plot(y,z,U_DSCPU,st=:surface,camera=(0,90))
# png("2D_mid_res")
