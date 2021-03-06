include("Tree.jl")


"""
Apply a simple affine transformation on each feature of the dataset to make the data suitable with the algorithm
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

"""
Used if the labels are String or everything else but Integer.
Create a new array with Integer labels and another array making the link between Integer and real labels.
"""
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


"""
Used to read a .txt format datafile
"""
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

"""
Used to read a .jl data file.\n
Example :\n
X=[...]
Y=[...]
K=...
"""
function read_X_Y_file(inputFile::String)
    include(inputFile)
end

"""
Save in a file the current data and labels.
"""
function save_X_Y(outputFile::String,X,Y)
    writer=open(outputFile,"w")
    print(writer,"X=")
    println(writer,X)
    println(writer,"Y=")
    println(writer,Y)
    close(writer)
end

function print_tree(T::Tree)
    D=T.D
    p=length(T.a[:,1])
    for k in 0:D-1
        for k2 in 1:div(2^(D-1),2^(D-k-1))
            for k3 in 1:2^(D-k-1)
                print("     ")
            end
            j=1
            while j<p+1 && T.a[j,2^k+k2-1]!=1
                j+=1
            end
            if j==p+1
                print(0)
            else
                print(j)
            end
            print("/",round(T.b[2^k+k2-1],digits=2))
        end
        println("")
    end
    for k in 1:2^D
        print(T.c[k])
        print("     ")
    end
end

function print_forest(forest::Array{Tree,1})
    n=length(forest)
    for i in 1:n
        print_tree(forest[i])
        println("")
    end
end


