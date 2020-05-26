using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Printf

N = 10
Δt = 0.1

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
# @show Matrix(A)

b = -ones(N, 1)
# @show Matrix(b)


function comp_piv_matrices(p,q,r)

    N = length(p)

    Id = sparse(1:N,1:N,ones(N),N,N)
    P = copy(Id)
    Q = copy(Id)

    for i = 1:N
        P[i,:] = Id[p[i],:]
        Q[:,i] = Id[:,q[i]]
    end

    R = sparse(1:N,1:N,r,N,N)

    return (P,Q,R)

end

#=
    Perform LU solve
=#
println("Perform Julia Native LU Factorization - Sparse")
F = lu(A)     # Call function once before timing
@time F = lu(A)
(P,Q,R) = comp_piv_matrices(F.p,F.q,F.Rs)
@assert F.L*F.U ≈ P*R*A*Q
@printf "norm(PRAQ-LU) = \x1b[31m %e \x1b[0m\n" norm(F.L*F.U-P*R*A*Q)
println("-----------")
println()

println("Perform Julia Native LU Solve - Sparse")
@time u = Q*(F.U\(F.L\(P*R*b)))
umod = A\b                      # Use Julia built in
@assert u ≈ umod
@printf "norm(u-umod) = \x1b[31m %e \x1b[0m\n" norm(u-umod)
println("-----------")
println()


#=
    Perform CG solve
=#
println("Perform Native Julia CG solve")
u = cg(A, b)            #call once
@time x = cg(A, b)
println("-----------")
println()
