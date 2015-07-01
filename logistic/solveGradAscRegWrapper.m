function testY = solveGradAscRegWrapper(trainX, trainY, testX)

    global learningRate;
    global permittedIterations;
    
    thetas = solveGradAscReg(trainX, trainY, learningRate, permittedIterations);
    classes = unique(double(trainY));
    testY = predict(testX, thetas, classes);
    
end
