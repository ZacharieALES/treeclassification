include("generation.jl")

using CPLEX
using JuMP

function find_Al_Ar_aux(x::Int64,Al::Array{Int64,1},Ar::Array{Int64,1})
    if x==1
        return(Al,Ar)
    elseif x%2==0
        new_x=div(x,2)
        find_Al_Ar_aux(new_x,append!(Al,new_x),Ar)
    else
        new_x=div(x-1,2)
        find_Al_Ar_aux(new_x,Al,append!(Ar,new_x))
    end
end

function find_Al_Ar(x::Int64)
    Al=zeros(Int64,0)
    Ar=zeros(Int64,0)
    find_Al_Ar_aux(x,Al,Ar)
end

function find_p(x::Int64)
    if x==1
        new_x=1
    elseif x%2==0
        new_x=div(x,2)
    else
        new_x=div(x-1,2)
    end
    return(new_x)
end 


function classification_tree_MIO(D::Int64,N_min::Int64,C::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,warm_start::Tree=nothing,H::Bool=false,alpha::Float64=0)
    n=length(Y)
    p=length(X[1])

    Yk=-ones(Int64,n,K)

    for i in 1:n
        Yk[i,Y[i]]=1
    end

    eps=ones(Float64,p)
    for i_1 in 1:n
        for i_2 in i_1:n
            for j in 1:p
                diff=abs(X[i_1,j]-X[i_2,j])
                if diff>0 && diff<eps[j]
                    eps[j]=diff
                end
            end
        end
    end

    eps_max=0
    for j in 1:p
        if eps[j]>eps_max
            eps_max=eps[j]
        end
    end

    Count_class=zeros(Int64,K)
    for i in 1:n
        Count_class[Y[i]]+=1
    end

    L_hat=max(Count_class)

    m=Model(with_optimizer(CPLEX.Optimizer))

    n_l=2^D #Leaves number
    n_b=2^D-1 #Branch nodes number

    @variable(m, L[1:n_l],Int)
    @variable(m, N[1:n_l],Int)
    @variable(m, N_k[1:K,1:n_l],Int)
    @variable(m, c[1:K,1:n_l],Bin)
    @variable(m, z[1:n,1:n_l],Bin)
    if H
        @variable(m, a[1:n_b,1:p],Bin)
    else
        @variable(m, a[1:n_b,1:p],Float)
        @variable(m, hat_a[1:n_b,1:p],Float)
        @variable(m, s[1:n_b,1:p],Bin)
    end
    @variable(m, b[1:n_b],Float)
    @variable(m, d[1:n_b],Bin)
    @variable(m, l[1:n_b],Bin)


    @constraint(m, [k in 1:K, t in 1:n_l], L[t] >= N[t]-N_k[k,t]-n*(1-c[k,t]))
    @constraint(m, [k in 1:K, t in 1:n_l], L[t] <= N[t]-N_k[k,t]+n*c[k,t])
    @constraint(m, [t in 1:n_l], L[t]>=0)
    @constraint(m, [k in 1:K, t in 1:n_l], N_k[k,t] == (1/2)*(sum((1+Yk[i,k])*z[i,t] for i in 1:n)))
    @constraint(m, [t in 1:n_l], N[t] == sum(z[it] for i in 1:n))
    @constraint(m, [t in 1:n_l], l[t] == sum(c[kt] for k in 1:K))
        
    mu=0.005

    for t in 1:n_l
        (Al,Ar)=find_Al_Ar(t+n_b)
        if H
            @constraint(m, [i in 1:n, m in Ar], sum(a[m,j]*x[i,j] for j in 1:p) >= b[m]-2*(1-z[i,t]))
            @constraint(m, [i in 1:n, m in Al], sum(a[m,j]*x[i,j] for j in 1:p) +mu <= b[m]+(2+mu)*(1-z[i,t]))
        else
            @constraint(m, [i in 1:n, m in Ar], sum(a[m,j]*x[i,j] for j in 1:p) >= b[m]-(1-z[i,t]))
            @constraint(m, [i in 1:n, m in Al], sum(a[m,j]*(x[i,j]+eps[j]) for j in 1:p) <= b[m]+(1+eps_max)*(1-z[i,t]))
        end
    end

    @constraint(m, [i in 1:n], sum(z[i,t] for t in 1:n_l)==1)
    @constraint(m, [i in 1:n, t in 1:n_l], z[i,t]<=l[t])
    @constraint(m, [t in 1:n_l], sum(z[i,t] for i in 1:n) >= N_min*l[t])
    @constraint(m, [t in 1:n_b], sum(a[t,j] for j in 1:p)==d[t])
    if H
        @constraint(m, [t in 1:n_b], -d[t] <= b[t])
    else
        @constraint(m, [t in 1:n_b], 0<= b[t])
    end
    @constraint(m, [t in 1:n_b], b[t]<= d[t])
    @constraint(m, [t in 2:n_b], d[t]<=d[find_p(t)])

    if H
        @constraint(m, [t in 1:n_b], sum(hat_a[j,t] for j in 1:p) <= d[t])
        @constraint(m,[j in 1:p,t in 1:n_b],a[j,t]<=hat_a[j,t])
        @constraint(m,[j in 1:p,t in 1:n_b],-a[j,t]<=hat_a[j,t])

        @constraint(m,[j in 1:p,t in 1:n_b],-s[j,t]<=a[j,t])
        @constraint(m,[j in 1:p,t in 1:n_b],a[j,t]<=s[j,t])
        @constraint(m,[j in 1:p,t in 1:n_b],s[j,t]<=d[t])
        @constraint(m, [t in 1:n_b], sum(s[j,t] for j in 1:p) >= d[t])
    end

    if warm_start !== nothing
        D_tree=warm_start.D
        n_b_tree=2^D_tree-1
        n_l_tree=2^D_tree
        set_start_value(a[1:n_b_tree,1:p],warm_start.a)
        set_start_value(b[1:n_b_tree],warm_start.b)
    end

    if H
        @objective(m, Min, (1/L_hat)*sum(L[t] for t in 1:n_l) + alpha*sum(sum(s[j,t] for j in 1:p) for t in 1:n_b))
    else
        @constraint(m, sum(d[t] for t in 1:n_b)<=C)
        @objective(m, Min, (1/L_hat)*sum(L[t] for t in 1:n_l))
    end

    final_c=zeros(Int64,n)
    n_l=2^D
    for t in 1:n_l
        k=1
        while value(m.c[k,t])!=1
            k+1
        end
        final_c[i]=k
    end

    return(Tree(D,value.(m.a),value.(m.b),final_c   ))
end

function score(T::Tree,X::Array{Float64,2},Y::Array{Int64,1})
    s=0
    D=T.D
    n=length(Y)
    p=length(X[1])
    max=2^D
    for i in 1:n
        if predict_class(T,X[i,:])!=Y[i]
            s+=1
        end
    end
    return(s)
end


function indice_best_tree(warm_start_list::Array{(Tree,Int64),1})
    i_min=1
    min=warm_start_list[1,2]
    for i in 2:length(warm_start_list)
        if warm_start_list[i,2]<min
            i_min=i
            min=warm_start_list[i,2]
        end
    end
    return(i_min)
end

function OCT(D_max::Int64,N_Min::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,H::Bool=false,alpha::Array{Float64,1})
    warm_start_list=[]
    n=length(Y)
    for D in 1:D_max
        for C in 1:2^D-1
            if warm_start_list==[]
                new_tree=classification_tree_MIO(D,N_Min,C,X,Y,K)
            else
                i=indice_best_tree(warm_start_list)
                new_tree=classification_tree_MIO(D,N_Min,C,X,Y,K,warm_start_list[i,1])
            end

            #create the c form used in Tree struct
            
            
            missclassification=score(new_tree,X,Y)
            append!(warm_start_list,(new_tree,missclassification))
        end
    end
    return(warm_start_list[indice_best_tree[warm_start_list]])
end
