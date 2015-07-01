function plotResults(companyData, trainX, trainY, trainDateOffset, testY, fittedCurve, predictedPrices)

    trainX = trainX(:,2);
    trainX = trainX + repmat(trainDateOffset, size(trainX));

    testDates = companyData(:,1);

    datesAxis = [trainX; testDates];
    pricesAxis = [trainY; testY];
    fittedCurveAxis = [fittedCurve; zeros(size(predictedPrices))];
    predictedPricesAxis = [zeros(size(fittedCurve)); predictedPrices];

    plot(datesAxis, pricesAxis, 'Color', 'black');
    hold on
    plot(datesAxis, fittedCurveAxis, 'Color', 'blue');
    hold on
    plot(datesAxis, predictedPricesAxis, 'Color', 'green');

end
