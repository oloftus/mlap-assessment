function testY = solveFminuncRegWrapper(trainX, trainY, testX)

    thetas = solveFminuncReg(trainX, trainY);
    classes = unique(double(trainY));
    testY = predict(testX, thetas, classes);
    
end
