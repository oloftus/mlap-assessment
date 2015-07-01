function yfit = solveFminuncWrapper(XTRAIN, ytrain, XTEST)

    thetas = solveFminunc(XTRAIN, ytrain);
    yfit = XTEST * thetas;
end
