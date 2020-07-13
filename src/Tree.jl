"""
Tree structure, used to store the result of OCT and predict labels.
"""
mutable struct Tree
    D::Int64
    a::Array{Float64,2}
    b::Array{Float64,1}
    c::Array{Int64,1}

    function Tree()
        return new()
    end

    
end

"""
Create a Tree instance.\n
Arguments :\n
    - D : the depth of the Tree
    - a and b : nodes' values used to choose the branch followed (a.x <b or not)
    - c : labels of the leaves
"""
function Tree(D::Int64,a::Array{Float64,2},b::Array{Float64,1},c::Array{Int64,1})
    this=Tree()
    this.D=D
    this.a=a
    this.b=b
    this.c=c
    return(this)
end

"""
Create a Tree with depth=0 used when there is no warm-up in the algorithm
"""
function null_Tree()
    this=Tree()
    this.D=0
    this.a=zeros(Float64,0,0)
    this.b=zeros(Float64,0)
    this.c=zeros(Int64,0)
    return(this)
end

"""
Return the label predicted for a single observation by a given tree
"""
function predict_one(T::Tree,x::Array{Float64,1})
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

"""
Return an array containing the labels predicted for several observations by a given tree.
"""
function predict(T::Tree,X::Array{Float64,2},need_leaf::Bool=false)
    n=length(X[:,1])
    p=length(X[1,:])
    max=2^T.D-1
    Y=zeros(Int64,n)
    leaves=zeros(Int64,n)
    for i in 1:n
        node=1
        while node<=max
            if sum(T.a[j,node]*X[i,j] for j in 1:p)<T.b[node]
                node=2*node
            else
                node=2*node+1
            end
        end
        Y[i]=T.c[node-max]
        leaves[i]=node-max
    end
    if need_leaf
        return(Y,leaves)
    else
        return(Y)
    end
end

"""
function score(Y1::Array{Int64,1},Y2::Array{Int64,1})
    return(sum(Y1[i]!=Y2[i] for i in 1:length(Y1)))
end
