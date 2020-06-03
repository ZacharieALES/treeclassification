include("io.jl")

function generate_Tree(D::Int64,p::Int64,K::Int64,H::Bool=false)
    n_l=2^D
    n_b=n_l-1
    if H
        a=rand(Float64,(p,n_b))
        for t in 1:n_b
            sum=sum(a[j,t] for j in 1:p)
            for j in 1:p
                a[j,t]=a[j,t]/sum
            end
        end   
    else
        a=zeros(Float64,(p,n_b))
        for t in 1:n_b
            j=rand(1:p)
            a[j,t]=1
        end
    end
    b=rand(Float64,n_b)
    c=rand(1:K,n_l)
    
    return(Tree(D,a,b,c))
end

function generate_X(n::Int64,p::Int64)
    X=rand(Float64,(n,p))
    return(X)
end

function predict_class(T::Tree,x::Array{Float64,1})
    p=length(x)
    max=2^T.D-1
    node=1
    while node<=max
        if sum(T.a[j,node]*x[j] for j in 1:p)<T.b[node]
            node=2*node
        else
            node=2*node+1
        end
    end
    return(T.c[node-max])
end

function generate_Y(T::Tree,X::Array{Float64,2})
    n=length(X[:,1])
    Y=zeros(Int64,n)
    for i in 1:n
        Y[i]=predict_class(T,X[i,:])
    end
    return(Y)
end

function generate_X_Y(D::Int64,p::Int64,K::Int64,n::Int64,H::Bool=false)
    T=generate_Tree(D,p,K,H)
    X=generate_X(n,p)
    Y=generate_Y(T,X)

    return(X,Y)
end