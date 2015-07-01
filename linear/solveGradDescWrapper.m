function yfit = solveGradDescWrapper(XTRAIN, ytrain, XTEST)

    global momentumFactor;
    global learningRates;
    
    thetas = solveGradDesc(XTRAIN, ytrain, momentumFactor, learningRates);
    yfit = XTEST * thetas;
end
