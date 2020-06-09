mutable struct Tree
    D::Int64
    a::Array{Float64,2}
    b::Array{Float64,1}
    c::Array{Int64,1}

    function Tree()
        return new()
    end
end

function Tree(D::Int64,a::Array{Float64,2},b::Array{Float64,1},c::Array{Int64,1})
    this=Tree()
    this.D=D
    this.a=a
    this.b=b
    this.c=c
    return(this)
end


function null_Tree()
    this=Tree()
    this.D=0
    this.a=zeros(Float64,0,0)
    this.b=zeros(Float64,0)
    this.c=zeros(Int64,0)
    return(this)
end