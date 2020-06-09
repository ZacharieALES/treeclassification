include("resolution.jl")

results=zeros(Float64,4)

read_X_Y_file("../data/testData.txt")

"""for D in 1:3
    T,result=classification_tree_MIO(D,5,2^D-1,X,Y,4)
    results[D]=result/2

    println("Pour D=", D," on obtient une erreur de classification de ", result/2)
end

println(results)"""

OCT(3,5,X,Y,4,true,[0.0, 2, 4, 6])