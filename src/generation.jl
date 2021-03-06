include("io.jl")

"""
Generate a random tree.\n
    - D : depth of the tree
    - p : number of features
    - K : number of possible labels
    - H : is the tree using uni-variate of multi-variate approach.
"""
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

"""
Generate a generate a random dataset
"""
function generate_X(n::Int64,p::Int64)
    X=rand(Float64,(n,p))
    return(X)
end

"""
Generate the labels associated to a given dataset X according to the tree T.
"""
function generate_Y(T::Tree,X::Array{Float64,2})
    return(predict(T,X))
end

"""
Generate a dataset X and the labels associated to each observation using a random tree.
"""
function generate_X_Y(D::Int64,p::Int64,K::Int64,n::Int64,H::Bool=false)
    T=generate_Tree(D,p,K,H)
    X=generate_X(n,p)
    Y=generate_Y(T,X)

    return(X,Y)
end

"""
Generate a boolean array of size n with a given proportion of true.\n 
This function is mainly used to determine which observation is in the training set and which is in the test set.
"""
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


"""
Generate a folder with severall files containing data and labels used to test the algorithm.\n
    - D : depth of the tree used to associate data and labels.
    - p : number of features for the observations.
    - K : number of different labels.
    - n : number of observations
    - nb_instance : number of files generated 
"""
function generate_data_set(datadir::String,D::Int64,p::Int64,K::Int64,n::Int64,nb_instance::Int64,H::Bool=false)
    for i in 1:nb_instance
        filename=datadir*"/instance_"*string(i)*"_D"*string(D)*"_p"*string(p)*"_K"*string(K)*"_n"*string(n)*".txt"
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

        print(writer,"K=")
        println(writer,K)
        close(writer)
    end
end

"""
Function used in the forest algorithm to separate the data in severall smaller datasets
"""
function create_sets(n::Int64,nb_set::Int64,n_by_set::Int64)
    sets=zeros(Bool,nb_set,n)
    count=zeros(Int64,nb_set)
    i=1
    while i<n
        indi=rand(1:nb_set)
        if count[indi]<n_by_set
            sets[indi,i]=true
            count[indi]+=1
            i+=1
        end
    end

    for i in 1:nb_set
        while count[i]<n_by_set
            indi=rand(1:n)
            if !sets[i,indi]
                sets[i,indi]=true
                count[i]+=1
            end
        end
    end
    return(sets)
end