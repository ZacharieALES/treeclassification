include("resolution.jl")

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