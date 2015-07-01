function testY = solveGradAscWrapper(trainX, trainY, testX)

    global learningRate;
    global momentumFactor;
    
    thetas = solveGradAsc(trainX, trainY, learningRate, momentumFactor);
    classes = unique(double(trainY));
    testY = predict(testX, thetas, classes);
    
end
