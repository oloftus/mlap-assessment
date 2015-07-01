function yfit = solveFminuncRegWrapper(XTRAIN, ytrain, XTEST)

    thetas = solveFminuncReg(XTRAIN, ytrain);
    yfit = XTEST * thetas;
end
