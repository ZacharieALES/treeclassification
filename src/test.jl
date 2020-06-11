include("resolution.jl")


function test_missclassification(X,Y,K::Int64,Dmax::Int64,need_scale::Bool=true,need_integer_labels::Bool=true,prop::Int64=25,H::Bool=false,alpha_array::Array{Float64,1}=[0.0])
    if need_scale
        X=zero_one_scaling(X)
    end
    if need_integer_labels
        Y,ref=create_integer_labels(Y)
    end

    n=length(Y)

    train,test=generate_sample(n,prop)

    tree=OCT(Dmax,round(Int64,2*n/100),X[train,:],Y[train],K,H,alpha_array)

    CART_tree=DecisionTreeClassifier(max_depth=Dmax)
    fit!(CART_tree,X[train,:],Y[train])
        

    return(score(predict(tree,X[test,:]),Y[test]),score(DecisionTree.predict(CART_tree,X[test,:]),Y[test]))
end


function compare_OCT_CART(nb_test::Int64,D,n,p,K,Dmax)
    cart=0
    oct=0

    for i in 1:nb_test
        X,Y=generate_X_Y(D,p,K,n)
        infos=test_missclassification(X,Y,K,Dmax,false,false)
        println("OCT : ",infos[1]," CART : ",infos[2])
        if infos[1]<infos[2]
            oct=oct+1
        elseif infos[2]<infos[1]
            cart=cart+1
        end
    end
    return(oct,cart)
end

println(compare_OCT_CART(20,3,150,3,4,2))


#X,Y=load_data("digits")

#X=zero_one_scaling(X)
#Y,ref=create_integer_labels(Y)

#println(test_missclassification(X,Y,10,2,false,false))

#n=length(Y)
#for i in 1:n
#    if predict_class(tree,X[i,:])!=Y[i]
#        println(i)
#    end
#end

