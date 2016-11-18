function [outAB] = makePossibleComb(Vecs)

D = length(Vecs);

outAB = [];
for d = D:-1:1
   
    outAB = pairVecMat(Vecs{d},outAB);
    
end

