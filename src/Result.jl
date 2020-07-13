"""
A structure used to store the result of the test in the data folder, containing the main informations.
Such as the time needed by each methods, the percentage of missclassification on train and test datasets.
"""
mutable struct Result
    OCT_D::Int64
    OCT_time::Float64
    OCT_train_miss::Float64
    OCT_test_miss::Float64
    CART_size::Int64
    CART_time::Float64
    CART_train_miss::Float64
    CART_test_miss::Float64
    limited_CART_time::Float64
    limited_CART_train_miss::Float64
    limited_CART_test_miss::Float64

    function Result()
        return new()
    end
end

function Result(OCT_D::Int64,OCT_time::Float64,OCT_train_miss::Float64,OCT_test_miss::Float64,CART_size::Int64,CART_time::Float64,
    CART_train_miss::Float64,CART_test_miss::Float64,limited_CART_time::Float64,limited_CART_train_miss::Float64,
    limited_CART_test_miss::Float64)
    this=Result()
    this.OCT_D=OCT_D
    this.OCT_time=OCT_time
    this.OCT_train_miss=OCT_train_miss
    this.OCT_test_miss=OCT_test_miss
    this.CART_size=CART_size
    this.CART_time=CART_time
    this.CART_train_miss=CART_train_miss
    this.CART_test_miss=CART_test_miss
    this.limited_CART_time=limited_CART_time
    this.limited_CART_train_miss=limited_CART_train_miss
    this.limited_CART_test_miss=limited_CART_test_miss
    return(this)
end

"""
A function used to compute and store the mean of each features for a result folder.
"""
function synthetic_res(datadir::String)
    k=0
    mean_OCT_time=0
    mean_OCT_train_miss=0
    mean_OCT_test_miss=0
    mean_CART_size=0
    mean_CART_time=0
    mean_CART_train_miss=0
    mean_CART_test_miss=0
    mean_limited_CART_time=0
    mean_limited_CART_train_miss=0
    mean_limited_CART_test_miss=0
    
    for file in filter(x->occursin(".txt", x), readdir(datadir))
        k+=1
        res=include(datadir*"/"*file)
        mean_OCT_time+=res.OCT_time
        mean_OCT_train_miss+=res.OCT_train_miss
        mean_OCT_test_miss+=res.OCT_test_miss
        mean_CART_size+=res.CART_size
        mean_CART_time+=res.CART_time
        mean_CART_train_miss+=res.CART_train_miss
        mean_CART_test_miss+=res.CART_test_miss
        mean_limited_CART_time+=res.limited_CART_time
        mean_limited_CART_train_miss+=res.limited_CART_train_miss
        mean_limited_CART_test_miss+=res.limited_CART_test_miss
    end

    writer=open(datadir*"/synthetic_res","w")
    println(writer,"#################################################")
    println(writer,"# Résultats synthétiques pour ce jeu de données #")
    println(writer,"#################################################")
    println(writer,"Temps moyen mis par OCT : ",mean_OCT_time/k)
    println(writer,"Pourcentage d'erreur de classification moyen (train) : ",mean_OCT_train_miss/k,"%")
    println(writer,"Pourcentage d'erreur de classification moyen (test) : ",mean_OCT_test_miss/k,"%")
    println(writer,"")
    println(writer,"Taille moyenne de CART : ",mean_CART_size/k)
    println(writer,"Temps moyen mis par CART : ",mean_CART_time/k)
    println(writer,"Pourcentage d'erreur de classification moyen (train) : ",mean_CART_train_miss/k,"%")
    println(writer,"Pourcentage d'erreur de classification moyen (test) : ",mean_CART_test_miss/k,"%")
    println(writer,"")
    println(writer,"Temps moyen mis par CART limité : ",mean_limited_CART_time/k)
    println(writer,"Pourcentage d'erreur de classification moyen (train) : ",mean_limited_CART_train_miss/k,"%")
    println(writer,"Pourcentage d'erreur de classification moyen (test) : ",mean_limited_CART_test_miss/k,"%")
    close(writer)
end