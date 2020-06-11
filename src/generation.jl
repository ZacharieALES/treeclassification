include("io.jl")

function generate_Tree(D::Int64,p::Int64,K::Int64,H::Bool=false)
    n_l=2^D
    n_b=n_l-1
    if H
        a=rand(Float64,(p,n_b))
        for t in 1:n_b
            somme=sum(a[j,t] for j in 1:p)
            for j in 1:p
                a[j,t]=a[j,t]/somme
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


function generate_Y(T::Tree,X::Array{Float64,2})
    return(predict(T,X))
end

function generate_X_Y(D::Int64,p::Int64,K::Int64,n::Int64,H::Bool=false)
    T=generate_Tree(D,p,K,H)
    X=generate_X(n,p)
    Y=generate_Y(T,X)

    return(X,Y)
end

function generate_sample(n::Int64,prop::Int64)
    nb=0
    train=ones(Bool,n)
    test=zeros(Bool,n)
    while nb<round(prop/100*n)
        i=rand(1:n)
        if train[i]
            train[i]=false
            test[i]=true
            nb=nb+1
        end
    end
    return(train,test)
end

function generate_data_set(D::Int64,p::Int64,K::Int64,n::Int64,nb_instance::Int64,H::Bool=false)
    for i in 1:nb_instance
        filename="../data/instance_"*string(i)*"_D"*string(D)*"_p"*string(p)*"_K"*string(K)*"_n"*string(n)
        writer=open(filename,"w")
        T=generate_Tree(D,p,K,H)
        X=generate_X(n,p)
        Y=generate_Y(T,X)

        print(writer,"T=")
        println(writer,T)

        print(writer,"X=")
        println(writer,X)

        print(writer,"Y=")
        println(writer,Y)
        close(writer)
    end
end