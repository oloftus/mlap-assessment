function testY = solveFminuncWrapper(trainX, trainY, testX)

    thetas = solveFminunc(trainX, trainY);
    classes = unique(double(trainY));
    testY = predict(testX, thetas, classes);
    
end
