include("generation.jl")
include("Result.jl")

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

"""
Function giving the Al and Ar index sets for a given node in a binary tree.
"""
function find_Al_Ar(x::Int64)
    Al=zeros(Int64,0)
    Ar=zeros(Int64,0)
    find_Al_Ar_aux(x,Al,Ar)
end

"""
Function giving the father of a given node in a binary tree.
"""
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

"""
Function solving the MIO problem.\n
Arguments :\n
    - D : the depth of the resulting tree 
    - N_min : the minimum number of observation attributed to a leaf.
    - X and Y : the data and labels
    - K : the number of labels.
Optionnal :\n
    - C/alpha : constants linked to the complexity of the tree
    - warm_start : You can give a warm_start tree as a starting point to seek the optimal solution
    - time_limit (in second)
    - H : is the approache uni or multi-variate.
Result :\n
    - The tree
    - The missclassification on the training set
    - The gap between the optimal solution and the solution found if the time limit was reached.
"""
function classification_tree_MIO(D::Int64,N_min::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,C::Int64=0,warm_start::Tree=null_Tree(),H::Bool=false,alpha::Float64=0.0,needZ::Bool=false,verbose::Bool=false,time_limit::Int64=-1)
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

    if time_limit!=-1
        set_time_limit_sec(model,time_limit)
    end

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
            @constraint(model,[i in 1:n,m in Al], sum(a[j,m]*(X[i,j]+eps[j]/2) for j in 1:p)<= b[m]+(1+eps_min)*(1-z[i,t]))
            @constraint(model, [i in 1:n, m in Al], z[i,t]<=d[m])
        end
    end


    #Warmstart part
    if warm_start.D !=0
        D_tree=warm_start.D
        n_b_tree=2^D_tree-1
        n_l_tree=2^D_tree
        for t in 1:n_b_tree
            set_start_value(b[t],warm_start.b[t])
            if warm_start.b[t]>0
                set_start_value(d[t],1)
            else
                set_start_value(d[t],0)
            end
            for j in 1:p
                set_start_value(a[j,t],warm_start.a[j,t])
            end
        end
        for t in n_b_tree+1:n_b
            set_start_value(b[t],0)
            set_start_value(d[t],0)
            for j in 1:p
                set_start_value(a[j,t],0)
            end
        end
        
        step=2^(D-D_tree)
        pred,leaves=predict(warm_start,X,true)

        real_Nkt=zeros(Int64,K,n_l)
        real_N=zeros(Int64,n_l)

        for t in 1:n_l

            for i in 1:n

                if leaves[i]==div(t,step)
                    set_start_value(z[i,t],1)
                    set_start_value(l[t],1)
                    real_Nkt[pred[i],t]+=1
                    real_N[t]+=1
                else
                    set_start_value(z[i,t],0)
                end
            end

            real_c=0
            if (t%step)==0 && warm_start.c[div(t,step)]!=0
                real_c=warm_start.c[div(t,step)]
                set_start_value(L[t],real_N[t]-real_Nkt[real_c,t])
            else
                set_start_value(L[t],0)
            end

            set_start_value(N[t],real_N[t])

            for k in 1:K
                if (t%step)==0 && k==real_c
                    set_start_value(c[k,t],1)
                else
                    set_start_value(c[k,t],0)
                end
                set_start_value(N_k[k,t],real_Nkt[k,t])
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

    gap=0

    if termination_status(model) != MOI.OPTIMAL
        gap=abs(JuMP.objective_bound(model) - JuMP.objective_value(model)) / JuMP.objective_value(model)
    end

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
        return(Tree(D,value.(a),value.(b),final_c),sum(value(L[t]) for t in 1:n_l),gap,value.(z))
    else
        return(Tree(D,value.(a),value.(b),final_c),sum(value(L[t]) for t in 1:n_l),gap)
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
    this_tree,miss,gap,z=classification_tree_MIO(1,1,X,Y,K,0,null_Tree(),true,0.0,true)
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


"""
Find a heuristic tree by applying the algorithm with a depth=1 recursively.\n 
It gives a good warm_start solution for the multivariate approach.
"""
function heuristic_OCT_H(D::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64)
    p=length(X[1,:])
    n_b=2^D-1
    current_tree=Tree(D,zeros(Float64,p,n_b),zeros(Float64,n_b),zeros(Int64,n_b+1))
    heuristic_OCT_H_aux(D,X,Y,K,1,current_tree)

    return(current_tree,score(predict(current_tree,X),Y))
end

"""
Optimal Classification Tree algorithm.\n
Find the tree that minimize the missclassification of the dataset for a given depth.\n 
Arguments :\n
    - D : the depth of the resulting tree 
    - N_min : the minimum number of observation attributed to a leaf.
    - X and Y : the data and labels
    - K : the number of labels.
"""
function OCT(D_max::Int64,N_Min::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,H::Bool=false,alpha_array::Array{Float64,1}=[0.0],need_gap::Bool=true,time_limit::Int64=-1)
    
    start=time()

    frac_time=0

    reason=4
    double_reason=reason*2

    if H
        frac_time=length(alpha_array)*reason*(reason^D_max-1)/(reason-1)
    else
        frac_time=double_reason*(double_reason^D_max-1)/(double_reason-1)-reason*(reason^D_max-1)/(reason-1)
    end
    
    warm_start_list=Tree[]
    miss_list=zeros(Int64,0)
    gaps=zeros(Float64,0)
    n=length(Y)
    for D in 1:D_max
        if H
            warm_start,miss=heuristic_OCT_H(D,X,Y,K)
            append!(warm_start_list,[warm_start])
            append!(miss_list,[miss])
            for alpha in alpha_array

                if time_limit==-1
                    allowed_time=-1
                else
                    remaining_time=time_limit-(time()-start)
                    allowed_time=ceil(Int64,remaining_time*reason^D/frac_time)
                end

                if warm_start_list==[]
                    new_tree,missclassification,gap=classification_tree_MIO(D,N_Min,X,Y,K,0,null_Tree(),true,alpha,false,false,allowed_time)
                else
                    i=indice_min(miss_list)
                    new_tree,missclassification,gap=classification_tree_MIO(D,N_Min,X,Y,K,0,warm_start_list[i],true,alpha,false,false,allowed_time)
                end

                append!(warm_start_list,[new_tree])
                append!(miss_list,[missclassification])
                append!(gaps,[gap])

                frac_time-=reason^D

            end

        else

            for C in 1:2^D-1

                if time_limit==-1
                    allowed_time=-1
                else
                    remaining_time=time_limit-(time()-start)
                    allowed_time=ceil(Int64,remaining_time*reason^D/frac_time)
                end


                if warm_start_list==[]
                    new_tree,missclassification,gap=classification_tree_MIO(D,N_Min,X,Y,K,C,null_Tree(),false,0.0,false,false,allowed_time)
                else
                    i=indice_min(miss_list)
                    new_tree,missclassification,gap=classification_tree_MIO(D,N_Min,X,Y,K,C,warm_start_list[i],false,0.0,false,false,allowed_time)
                end
                
                append!(warm_start_list,[new_tree])
                append!(miss_list,[round(Int64,missclassification)])
                append!(gaps,[gap])

                frac_time-=reason^D
            end

        end
    end
    best_i=indice_min(miss_list)

    if need_gap
        return(warm_start_list[best_i],gaps[best_i])
    else
        return(warm_start_list[best_i])
    end
end



"""
First version of a Forest version of OCT dividing the training set in several sets. The objective was to reduce the computing time.
The algorithm create one tree for each subset to create a forest.
"""
function OCT_forest(D_max::Int64,N_Min::Int64,X::Array{Float64,2},Y::Array{Int64,1},K::Int64,H::Bool=false,alpha_array=[0.0],nb_tree::Int64=1,percentage_for_one::Int64=-1,time_limit::Int64=-1)
    n=length(X[:,1])
    p=length(X[1,:])
    n_one_tree=0

    start=time()

    if percentage_for_one==-1
        n_one_tree=div(n,nb_tree)
    else
        n_one_tree=floor(Int64,n*percentage_for_one/100)
    end

    sets=create_sets(n,nb_tree,n_one_tree)

    trees=Tree[]

    for i in 1:nb_tree
        if time_limit==-1
            allowed_time=-1
        else
            remaining_time=time_limit-(time()-start)
            allowed_time=ceil(Int64,remaining_time/(nb_tree-i+1))
        end

        new_tree=OCT(D_max,N_Min,X[sets[i,:],:],Y[sets[i,:]],K,H,alpha_array,false,allowed_time)
        append!(trees,[new_tree])

    end

    return(trees)
end

"""
A prediction function for forest algorithm that keep the most predicted label by the forest for each observation.
""" 
function predict_forest(forest::Array{Tree,1},X::Array{Float64,2},K::Int64)
    n=length(X[:,1])
    Y=zeros(Int64,n)

    nb_tree=length(forest)
    
    
    for i in 1:n
        count=zeros(Int64,K)
        for j in 1:nb_tree
            indi=predict_one(forest[j],X[i,:])
            if indi>0
                count[indi]+=1
            end
        end

        max=count[1]
        j_max=1
        for j in 2:K
            if max<count[j]
                max=count[j]
                j_max=j
            end
        end

        Y[i]=j_max
    end
    return(Y)
end


"""
The second version of forest algorithm that add the creation of the different trees to the MIO problem thanks to new variables and constraints.
The aim is to create trees that are close one to another.
"""
function classification_forest_MIO(D::Int64,N_min::Int64,X::Array{Float64,3},Y::Array{Int64,2},K::Int64,C::Int64=0,beta::Float64=0.0,time_limit::Int64=-1)
    nb_tree=length(Y[:,1])

    n=length(Y[1,:])
    p=length(X[1,1,:])
    Yk=-ones(Int64,nb_tree,n,K)

    for q in 1:nb_tree
        for i in 1:n
            Yk[q,i,Y[q,i]]=1
        end
    end

    eps=ones(Float64,nb_tree,p)
    for q in 1:nb_tree
        for i_1 in 1:n
            for i_2 in i_1+1:n
                for j in 1:p
                    diff=abs(X[q,i_1,j]-X[q,i_2,j])
                    if diff>0 && diff<eps[q,j]
                        eps[q,j]=diff
                    end
                end
            end
        end
    end

    eps_min=ones(Float64,nb_tree)
    for q in 1:nb_tree
        for j in 1:p
            if eps[q,j]<eps_min[q]
                eps_min[q]=eps[q,j]
            end
        end
    end

    Count_class=zeros(Int64,K)
    for q in 1:nb_tree
        for i in 1:n
            Count_class[Y[q,i]]+=1
        end
    end

    L_hat=Count_class[1]
    for k in 2:K
        if Count_class[k]>L_hat
            L_hat=Count_class[k]
        end
    end 

    model=Model(CPLEX.Optimizer)

    set_silent(model)

    if time_limit!=-1
        set_time_limit_sec(model,time_limit)
    end

    n_l=2^D #Leaves number
    n_b=2^D-1 #Branch nodes number

    # Variables d'origine

    @variable(model, L[1:nb_tree,1:n_l],Int)
    @variable(model, N[1:nb_tree,1:n_l],Int)
    @variable(model, N_k[1:nb_tree,1:K,1:n_l],Int)
    @variable(model, c[1:nb_tree,1:K,1:n_l],Bin)
    @variable(model, z[1:nb_tree,1:n,1:n_l],Bin)
    @variable(model, a[1:nb_tree,1:p,1:n_b],Bin)
    @variable(model, b[1:nb_tree,1:n_b]) 
    @variable(model, d[1:nb_tree,1:n_b],Bin)
    @variable(model, l[1:nb_tree,1:n_l],Bin)

    # Variables supplémentaires
    @variable(model, A[1:nb_tree, 1:nb_tree,1:p,1:n_b],Bin)

    # Contraintes d'origine

    @constraint(model, [q in 1:nb_tree,i in 1:n], sum(z[q,i,t] for t in 1:n_l)==1)
    @constraint(model, [q in 1:nb_tree,i in 1:n, t in 1:n_l], z[q,i,t]<=l[q,t])
    @constraint(model, [q in 1:nb_tree,t in 1:n_l], sum(z[q,i,t] for i in 1:n) >= N_min*l[q,t])
    @constraint(model, [q in 1:nb_tree,t in 1:n_b], sum(a[q,j,t] for j in 1:p)==d[q,t]) 
    @constraint(model, [q in 1:nb_tree,t in 1:n_b], 0<= b[q,t])
    @constraint(model, [q in 1:nb_tree,t in 1:n_b], b[q,t]<= d[q,t])
    @constraint(model, [q in 1:nb_tree,t in 2:n_b], d[q,t]<=d[q,find_p(t)])
    @constraint(model, [q in 1:nb_tree,k in 1:K, t in 1:n_l], L[q,t] >= N[q,t]-N_k[q,k,t]-n*(1-c[q,k,t]))
    @constraint(model, [q in 1:nb_tree,k in 1:K, t in 1:n_l], L[q,t] <= N[q,t]-N_k[q,k,t]+n*c[q,k,t])
    @constraint(model, [q in 1:nb_tree,t in 1:n_l], L[q,t]>=0)
    @constraint(model, [q in 1:nb_tree,k in 1:K, t in 1:n_l], N_k[q,k,t] == (1/2)*(sum((1+Yk[q,i,k])*z[q,i,t] for i in 1:n)))
    @constraint(model, [q in 1:nb_tree,t in 1:n_l], N[q,t] == sum(z[q,i,t] for i in 1:n))
    @constraint(model, [q in 1:nb_tree,t in 1:n_l], l[q,t] == sum(c[q,k,t] for k in 1:K))
    for t in 1:n_l
        (Al,Ar)=find_Al_Ar(t+n_b)
        @constraint(model,[q in 1:nb_tree,i in 1:n,m in Ar], sum(a[q,j,m]*X[q,i,j] for j in 1:p) >= b[q,m]-(1-z[q,i,t]))
        @constraint(model,[q in 1:nb_tree,i in 1:n,m in Al], sum(a[q,j,m]*(X[q,i,j]+eps[q,j]/2) for j in 1:p)<= b[q,m]+(1+eps_min[q])*(1-z[q,i,t]))
        @constraint(model, [q in 1:nb_tree,i in 1:n, m in Al], z[q,i,t]<=d[q,m])
    end
    @constraint(model,[q in 1:nb_tree] , sum(d[q,t] for t in 1:n_b)<=C)
    
    # Contraintes supplémentaires

    @constraint(model, [q1 in 1:nb_tree, q2 in 1:nb_tree, j in 1:p, t in 1:n_b], a[q1,j,t]-a[q2,j,t] <= A[q1,q2,j,t])
    @constraint(model, [q1 in 1:nb_tree, q2 in 1:nb_tree, j in 1:p, t in 1:n_b], -a[q1,j,t]+a[q2,j,t] <= A[q1,q2,j,t])
    @constraint(model, [q1 in 1:nb_tree, q2 in 1:nb_tree, j in 1:p, t in 1:n_b], a[q1,j,t]+a[q2,j,t]+A[q1,q2,j,t]<=2)
    @constraint(model, [q1 in 1:nb_tree, q2 in 1:nb_tree, j in 1:p, t in 1:n_b], a[q1,j,t]+a[q2,j,t] >= A[q1,q2,j,t])

    # Objectifs (modifié par rapport à l'objectif d'origine)

    @objective(model, Min, (1/L_hat)*sum(sum(L[q,t] for t in 1:n_l) for q in 1:nb_tree)+ beta/nb_tree*sum(sum(sum(sum(A[q1,q2,j,t] for j in 1:p) for t in 1:n_b) for q2 in q1+1:nb_tree)  for q1 in 1:nb_tree))
    
    optimize!(model)

    final_c=zeros(Int64,nb_tree,n_l)
    n_l=2^D
    for q in 1:nb_tree
        for t in 1:n_l
            if round(value(l[q,t]))==1
                k=1
                while round(value(c[q,k,t]))!=1
                    k=k+1
                end
                final_c[q,t]=k
            else
                final_c[q,t]=0
            end
        end
    end

    forest=Tree[]
    for i in 1:nb_tree
        append!(forest,[Tree(D,value.(a[i,:,:]),value.(b[i,:]),final_c[i,:])])
    end
    return(forest)


end