include("generation.jl")

using CPLEX
using JuMP

using DecisionTree

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


function classification_tree_MIO(D::Int64,N_min::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,C::Int64=0,warm_start::Tree=null_Tree(),H::Bool=false,alpha::Float64=0.0,needZ::Bool=false,verbose::Bool=false)
    n=length(Y)
    p=length(X[1,:])
    Yk=-ones(Int64,n,K)

    for i in 1:n
        Yk[i,Y[i]]=1
    end

    if verbose
        println("MIO initialisation")
    end

    eps=ones(Float64,p)
    for i_1 in 1:n
        for i_2 in i_1+1:n
            for j in 1:p
                diff=abs(X[i_1,j]-X[i_2,j])
                if diff>0 && diff<eps[j]
                    eps[j]=diff
                end
            end
        end
    end


    eps_min=1
    for j in 1:p
        if eps[j]<eps_min
            eps_min=eps[j]
        end
    end

    Count_class=zeros(Int64,K)
    for i in 1:n
        Count_class[Y[i]]+=1
    end

    
    L_hat=Count_class[1]
    for k in 2:K
        if Count_class[k]>L_hat
            L_hat=Count_class[k]
        end
    end 

    model=Model(CPLEX.Optimizer)

    if !verbose
        set_silent(model)
    end

    
    set_time_limit_sec(model,120*D)

    n_l=2^D #Leaves number
    n_b=2^D-1 #Branch nodes number

    @variable(model, L[1:n_l],Int)
    @variable(model, N[1:n_l],Int)
    @variable(model, N_k[1:K,1:n_l],Int)
    @variable(model, c[1:K,1:n_l],Bin)
    @variable(model, z[1:n,1:n_l],Bin)
    if H
        @variable(model, a[1:p,1:n_b])
        @variable(model, hat_a[1:p,1:n_b])
        @variable(model, s[1:p,1:n_b],Bin)
    else
        @variable(model, a[1:p,1:n_b],Bin)
    end
    @variable(model, b[1:n_b])  #vérifier que c'est bien un float puis faire pareil pour les autres
    @variable(model, d[1:n_b],Bin)
    @variable(model, l[1:n_l],Bin)


    @constraint(model, [i in 1:n], sum(z[i,t] for t in 1:n_l)==1)
    @constraint(model, [i in 1:n, t in 1:n_l], z[i,t]<=l[t])
    @constraint(model, [t in 1:n_l], sum(z[i,t] for i in 1:n) >= N_min*l[t])
    @constraint(model, [t in 1:n_b], sum(a[j,t] for j in 1:p)==d[t])  
    if H
        @constraint(model, [t in 1:n_b], -d[t] <= b[t])
    else
        @constraint(model, [t in 1:n_b], 0<= b[t])
    end
    @constraint(model, [t in 1:n_b], b[t]<= d[t])
    @constraint(model, [t in 2:n_b], d[t]<=d[find_p(t)])

    if H
        @constraint(model, [t in 1:n_b], sum(hat_a[j,t] for j in 1:p) <= d[t])
        @constraint(model,[j in 1:p,t in 1:n_b],a[j,t]<=hat_a[j,t])
        @constraint(model,[j in 1:p,t in 1:n_b],-a[j,t]<=hat_a[j,t])

        @constraint(model,[j in 1:p,t in 1:n_b],-s[j,t]<=a[j,t])
        @constraint(model,[j in 1:p,t in 1:n_b],a[j,t]<=s[j,t])
        @constraint(model,[j in 1:p,t in 1:n_b],s[j,t]<=d[t])
        @constraint(model, [t in 1:n_b], sum(s[j,t] for j in 1:p) >= d[t])
    end


    @constraint(model, [k in 1:K, t in 1:n_l], L[t] >= N[t]-N_k[k,t]-n*(1-c[k,t]))
    @constraint(model, [k in 1:K, t in 1:n_l], L[t] <= N[t]-N_k[k,t]+n*c[k,t])
    @constraint(model, [t in 1:n_l], L[t]>=0)
    @constraint(model, [k in 1:K, t in 1:n_l], N_k[k,t] == (1/2)*(sum((1+Yk[i,k])*z[i,t] for i in 1:n)))
    @constraint(model, [t in 1:n_l], N[t] == sum(z[i,t] for i in 1:n))
    @constraint(model, [t in 1:n_l], l[t] == sum(c[k,t] for k in 1:K))
        
    mu=0.005

    for t in 1:n_l
        (Al,Ar)=find_Al_Ar(t+n_b)
        
        if H
            @constraint(model, [i in 1:n,m in Ar], sum(a[j,m]*X[i,j] for j in 1:p) >= b[m]-2*(1-z[i,t]))
            @constraint(model, [i in 1:n,m in Al], sum(a[j,m]*X[i,j] for j in 1:p) +mu <= b[m]+(2+mu)*(1-z[i,t]))
        else
            @constraint(model,[i in 1:n,m in Ar], sum(a[j,m]*X[i,j] for j in 1:p) >= b[m]-(1-z[i,t]))
            @constraint(model,[i in 1:n,m in Al], sum(a[j,m]*(X[i,j]) for j in 1:p) + eps_min <= b[m]+(1+eps_min)*(1-z[i,t]))
        end
    end



    if warm_start.D !=0
        D_tree=warm_start.D
        n_b_tree=2^D_tree-1
        n_l_tree=2^D_tree
        for t in 1:n_b_tree
            set_start_value(b[t],warm_start.b[t])
            for j in 1:p
                set_start_value(a[j,t],warm_start.a[j,t])
            end
        end
    end

    if H
        @objective(model, Min, (1/L_hat)*sum(L[t] for t in 1:n_l) + alpha*sum(sum(s[j,t] for j in 1:p) for t in 1:n_b))
    else
        @constraint(model, sum(d[t] for t in 1:n_b)<=C)
        @objective(model, Min, (1/L_hat)*sum(L[t] for t in 1:n_l))
    end

    if verbose
        println("Solving ...")
    end

    optimize!(model)

    final_c=zeros(Int64,n_l)
    n_l=2^D
    for t in 1:n_l
        if round(value(l[t]))==1
            k=1
            while round(value(c[k,t]))!=1
                k=k+1
            end
            final_c[t]=k
        else
            final_c[t]=0
        end
    end

    

    #return(model)
    if needZ
        return(Tree(D,value.(a),value.(b),final_c),sum(value(L[t]) for t in 1:n_l),value.(z))
    else
        return(Tree(D,value.(a),value.(b),final_c),sum(value(L[t]) for t in 1:n_l))
    end

end



function indice_min(liste)
    i_min=1
    min=liste[1]
    for i in 2:length(liste)
        if liste[i]<min
            i_min=i
            min=liste[i]
        end
    end
    return(i_min)
end

function heuristic_OCT_H_aux(D::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,step::Int64,currentTree::Tree)
    max=2^D-1
    this_tree,miss,z=classification_tree_MIO(1,1,X,Y,K,0,null_Tree(),true,0.0,true)
    currentTree.a[:,step]=this_tree.a[:,1]
    currentTree.b[step]=this_tree.b[1]
    if 2*step<max
        left=Bool[]
        right=Bool[]
        nb_left=0
        for i in 1:length(Y)
            nb_left=nb_left+z[i,1]
            append!(left,z[i,1])
            append!(right,z[i,2])
        end
        if nb_left>0
            heuristic_OCT_H_aux(D,X[left,:],Y[left],K,2*step,currentTree) 
        end
        if nb_left<length(Y)
            heuristic_OCT_H_aux(D,X[right,:],Y[right],K,2*step+1,currentTree)
        end
    else
        currentTree.c[2*step-max]=this_tree.c[1]
        currentTree.c[2*step-max+1]=this_tree.c[2]
    end

end

function heuristic_OCT_H(D::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64)
    p=length(X[1,:])
    n_b=2^D-1
    current_tree=Tree(D,zeros(Float64,p,n_b),zeros(Float64,n_b),zeros(Int64,n_b+1))
    heuristic_OCT_H_aux(D,X,Y,K,1,current_tree)

    return(current_tree,score(predict(current_tree,X),Y))
end

function OCT(D_max::Int64,N_Min::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,H::Bool=false,alpha_array::Array{Float64,1}=[0.0])
    warm_start_list=Tree[]
    miss_list=zeros(Int64,0)
    n=length(Y)
    for D in 1:D_max
        if H
            warm_start,miss=heuristic_OCT_H(D,X,Y,K)
            append!(warm_start_list,[warm_start])
            append!(miss_list,[miss])
            for alpha in alpha_array
                if warm_start_list==[]
                    new_tree,missclassification=classification_tree_MIO(D,N_Min,X,Y,K,0,null_Tree(),true,alpha)
                else
                    i=indice_min(miss_list)
                    new_tree,missclassification=classification_tree_MIO(D,N_Min,X,Y,K,0,warm_start_list[i],true,alpha)
                end

                append!(warm_start_list,[new_tree])
                append!(miss_list,[missclassification])

            end

        else

            for C in 1:2^D-1

                if warm_start_list==[]
                    new_tree,missclassification=classification_tree_MIO(D,N_Min,X,Y,K,C)
                else
                    i=indice_min(miss_list)
                    new_tree,missclassification=classification_tree_MIO(D,N_Min,X,Y,K,C,warm_start_list[i])
                end
                            
                append!(warm_start_list,[new_tree])
                append!(miss_list,[round(Int64,missclassification)])
            end

        end
    end
    return(warm_start_list[indice_min(miss_list)])
end
