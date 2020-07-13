include("resolution.jl")

using MultivariateStats
using Random

function solveDataSet(datadir::String,prop::Int64,Dmax,H=false,alpha_array=[0.0],time_limit::Int64=-1)
    resFolder="../res"
    dataFolder="../data"

    cart=0
    oct=0

    for file in filter(x->occursin(".txt", x), readdir(dataFolder*"/"*datadir))  
        
        println("-- Resolution of ", file)

        include(dataFolder*"/"*datadir*"/"*file)

        n=length(Y)

        train,test=generate_sample(n,prop)

        OCT_time=time()

        tree,gap=OCT(Dmax,1,X[train,:],Y[train],K,H,alpha_array,false,time_limit)

        OCT_time=time()-OCT_time
        OCT_train=score(predict(tree,X[train,:]),Y[train])
        OCT_test=score(predict(tree,X[test,:]),Y[test])
        
        CART_time=time()
    
        CART_tree=DecisionTreeClassifier()
        fit!(CART_tree,X[train,:],Y[train])

        CART_size=DecisionTree.depth(CART_tree)

        CART_time=time()-CART_time
        CART_train=score(DecisionTree.predict(CART_tree,X[train,:]),Y[train])
        CART_test=score(DecisionTree.predict(CART_tree,X[test,:]),Y[test])

        limited_time=time()
        
        limited_tree=DecisionTreeClassifier(max_depth=Dmax)
        fit!(limited_tree,X[train,:],Y[train])

        limited_time=time()-limited_time

        limited_train=score(DecisionTree.predict(limited_tree,X[train,:]),Y[train])
        limited_test=score(DecisionTree.predict(limited_tree,X[test,:]),Y[test])

        factor=(100-prop)/100

        res=Result(Dmax,OCT_time,OCT_train*100/(n*factor),OCT_test*100/(n*(1-factor)),CART_size,CART_time,CART_train*100/(n*factor),CART_test*100/(n*(1-factor)),limited_time,limited_train*100/(n*factor),limited_test*100/(n*(1-factor)))


        if !isdir(resFolder*"/"*datadir)
            mkdir(resFolder*"/"*datadir)
        end

        writer=open(resFolder*"/"*datadir*"/"*file,"w")
        
        println(writer,res)

        if gap!=0
            println(writer,"# Gap = ",gap)
        end

        close(writer)

    end

end

function solveAll(D::Int64=2,time_limit::Int64=-1,H::Bool=false)
    for dir in readdir("../data")
        if isdir("../data/"*dir)
            println("Currently in folder : ", dir)
            solveDataSet(dir,25,D,H,[0.0],time_limit)
            synthetic_res("../res/"*dir)
        end
    end
end

function test_forest()

    datadir="../data/real_world"
    prop=25
    Dmax=3
    Nmin=1
    time_limit=200
    nb_tree=3
    percentage=40

    for file in filter(x->occursin(".txt", x), readdir(datadir))
        println("-- Resolution of ", file)

        include(datadir*"/"*file)

        n=length(Y)

        train,test=generate_sample(n,prop)


        OCT_time=time()

        tree=OCT(Dmax,Nmin,X[train,:],Y[train],K,false,[0.0],false,time_limit)

        OCT_time=time()-OCT_time
        OCT_train=score(predict(tree,X[train,:]),Y[train])
        OCT_test=score(predict(tree,X[test,:]),Y[test])

        println("## OCT ## Time : ",OCT_time,", erreur train/test : ",OCT_train,"/",OCT_test)

        forest_time=time()

        forest=OCT_forest(Dmax,Nmin,X[train,:],Y[train],K,false,[0.0],nb_tree,percentage,time_limit)
        
        forest_time=time()-forest_time
        forest_train=score(predict_forest(forest,X[train,:],K),Y[train])
        forest_test=score(predict_forest(forest,X[test,:],K),Y[test])

        println("## Forest ## Time : ",forest_time,", erreur train/test : ",forest_train,"/",forest_test)
    end  
end

function test_ACP()

    datadir="../data/real_world"
    prop=25
    Dmax=3
    Nmin=1
    time_limit=200


    for file in filter(x->occursin(".txt", x), readdir(datadir))
        println("-- Resolution of ", file)

        include(datadir*"/"*file)

        tX=transpose(X)
        M=fit(PCA, tX, maxoutdim=2)

        newX1=transpose(transform(M,tX))

        newX1=zero_one_scaling(newX1)

        M=fit(PCA, tX, maxoutdim=4)

        newX2=transpose(transform(M,tX))

        newX2=zero_one_scaling(newX2)

        n=length(Y)

        train,test=generate_sample(n,prop)

        OCT_time=time()

        tree=OCT(Dmax,Nmin,X[train,:],Y[train],K,false,[0.0],false,time_limit)

        OCT_time=time()-OCT_time
        OCT_train=score(predict(tree,X[train,:]),Y[train])
        OCT_test=score(predict(tree,X[test,:]),Y[test])

        println("## OCT ## Time : ",OCT_time,", erreur train/test : ",OCT_train,"/",OCT_test)

        acp_time=time()

        acp=OCT(Dmax,Nmin,newX1[train,:],Y[train],K,false,[0.0],false,time_limit)

        acp_time=time()-acp_time
        acp_train=score(predict(tree,newX1[train,:]),Y[train])
        acp_test=score(predict(tree,newX1[test,:]),Y[test])



        println("## ACP 2 composantes ## Time : ",acp_time,", erreur train/test : ",acp_train,"/",acp_test)

        acp_time=time()

        acp=OCT(Dmax,Nmin,newX2[train,:],Y[train],K,false,[0.0],false,time_limit)

        acp_time=time()-acp_time
        acp_train=score(predict(tree,newX2[train,:]),Y[train])
        acp_test=score(predict(tree,newX2[test,:]),Y[test])



        println("## ACP 3 composantes ## Time : ",acp_time,", erreur train/test : ",acp_train,"/",acp_test)
    end  
end

function test_new_forest()
    include("../data/real_world/iris.txt")
    train1=[3*i+1 for i in 0:49]
    train2=[3*i+2 for i in 0:49]
    test=[3*i+3 for i in 0:49]

    trainglobal=vcat(train1,train2)

    tps=time()
    OCT_tree,sc,gap=classification_tree_MIO(3,1,X[trainglobal,:],Y[trainglobal],K,7)
    tps=time()-tps

    tree_train=score(predict(OCT_tree,X[trainglobal,:]),Y[trainglobal])
    tree_test=score(predict(OCT_tree,X[test,:]),Y[test])

    println("# OCT #, train/test : ",tree_train,"/",tree_test,", time : ",tps)

    X_forest=zeros(Float64,2,50,4)
    X_forest[1,:,:]=X[train1,:]
    X_forest[2,:,:]=X[train2,:]

    Y_forest=zeros(Int64,2,50)
    Y_forest[1,:]=Y[train1]
    Y_forest[2,:]=Y[train2]

    tps=time()
    forest=classification_forest_MIO(3,1,X_forest,Y_forest,K,7,0.5)
    tps=time()-tps

    forest_train=score(predict_forest(forest,X[trainglobal,:],K),Y[trainglobal])
    forest_test=score(predict_forest(forest,X[test,:],K),Y[test])

    println("# Forest #, train/test : ",forest_train,"/",forest_test,", time : ",tps)


end
