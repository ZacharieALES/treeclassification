"""
Read a data file and create an array X containing the features and Y containing the class labels
"""

function readDataFile(inputFile::String,header::Bool=true,separator::Char=" ")
    dataFile=open(inputFile)
    data=readlines(dataFile)
    close(dataFile)

    i_min=1
    if header
        i_min=2
    end

    n=length(data)+1-i_min
    p=length(split(data[i_min],separator))-1
    X=zeros(Float64,n,p)
    x_min=zeros(Float64,p)
    x_max=zeros(Float64,p)
    Y=zeros(Int64,n)

    for i in 1:n
        line=split(data[i+i_min-1],separator)
        for j in 1:p
            X[i,j]=parse(Float64,line[j])
            if (i==i_min || line[j]>x_max[j])
                x_max[j]=X[i,j]
            end
            if (i==i_min || line[j]<x_max[j])
                x_min[j]=X[i,j]
            end
        end
        Y[i]=parse(Int64,line[p+1])
    end

    #Scaling the data in [0,1]
    for j in 1:p
        m=1/(x_max[j]-x_min[j])
        p=x_min[j]
        for i in 1:n
            X[i,j]=m*(X[i,j]-p)
        end
    end


    return(X,Y)
end


mutable struct Tree
    D::Int64
    a::Array{Float64,2}
    b::Array{Float64,1}
    c::Array{Int64,1}
    missclassification::Int64

    function Tree()
        return new()
    end
end

function Tree(D::Int64,a::Array{Float64,2},b::Array{Float64,1},c::Array{Int64,1},missclassification::Int64)
    this=Tree()
    this.D=D
    this.a=a
    this.b=b
    this.c=c
    this.missclassification=missclassification
    return(this)
end