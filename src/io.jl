include("Tree.jl")
"""
Read a data file and create an array X containing the features and Y containing the class labels
"""

function zero_one_scaling(X)
    n=length(X[:,1])
    p=length(X[1,:])
    newX=zeros(Float64,n,p)
    x_min=zeros(Float64,p)
    x_max=zeros(Float64,p)
    for i in 1:n
        for j in 1:p
            if (i==1 || X[i,j]>x_max[j])
                x_max[j]=X[i,j]
            end
            if (i==1 || X[i,j]<x_min[j])
                x_min[j]=X[i,j]
            end
        end
    end

    #Scaling the data in [0,1]
    indi=zeros(Bool,p)
    for j in 1:p
        if (x_max[j]-x_min[j])!=0
            m=1/(x_max[j]-x_min[j])
            p=x_min[j]
            for i in 1:n
                newX[i,j]=m*(X[i,j]-p)
            end
            indi[j]=true
        end
    end
    
    return(newX[:,indi])
end

function create_integer_labels(Y::Array{Any,1})
    n=length(Y[:,1])
    labels_ref=[]
    labels=zeros(Int64,n)
    for i in 1:n
        k=1
        k_max=length(labels_ref)+1
        while k<k_max && labels_ref[k,:]!=Y[i,:]
            k=k+1
        end
        if k==k_max
            append!(labels_ref,Y[i,:])
        end
        labels[i]=k
    end
    return(labels,labels_ref)
end



function readDataFile(inputFile::String,start::Int64,column::Array{Bool,1},labels::Int64=-1,separator::Char=' ')
    dataFile=open(inputFile)
    data=readlines(dataFile)
    close(dataFile)

    i_min=start

    n=length(data)+1-i_min
    p=sum(column[i] for i in 1:length(column))
    X=Array{Any,2}(nothing,n,p)
    x_min=Array{Any,1}(nothing,p)
    x_max=Array{Any,1}(nothing,p)
    Y=Array{Any,1}(nothing,n)

    if labels==-1
        labels=p+1
    end

    for i in 1:n
        line=split(data[i+i_min-1],separator,keepempty=false)
        j=1
        real_j=1
        while j<=p
            if column[real_j]
                X[i,j]=parse(Float64,line[real_j])
                if (i==i_min || X[i,j]>x_max[j])
                    x_max[j]=X[i,j]
                end
                if (i==i_min || X[i,j]<x_min[j])
                    x_min[j]=X[i,j]
                end
                j+=1
            end
            real_j+=1
        end
        Y[i]=line[labels]
    end



    return(X,Y)
end

function read_X_Y_file(inputFile::String)
    include(inputFile)
end

function save_X_Y(outputFile::String,X,Y)
    writer=open(outputFile,"w")
    print(writer,"X=")
    println(writer,X)
    println(writer,"Y=")
    println(writer,Y)
    close(writer)
end


