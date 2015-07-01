function [mseNoReg, thetasNoReg, mseReg, thetasReg, lambd, mseGradDesc, thetasGradDesc] = main()

    %% Parameters

    numKFolds = 5;
    companyDataFileName = 'data/company_stock_price.csv';
    sectorDataFileName = 'data/sector_stock_price.csv';
    endDate = datenum('22/10/2009', 'dd/mm/yyyy');
    knotPoints = [0, 10, 30, 80, 130];
    
    momentumFactor = 1.1;
    
    onesSelect                           = 1;
    datesSelect                          = 1;
    companyPricesMinus1DaySelect         = 1;
    dayNumSelect                         = 1;
    companyVolumesMinus1DaySelect        = 0;
    sectorVolumesMinus1DaySelect         = 0;
    companyPriceFiveDayAvgGradientSelect = 1;
    sectorPricesMinus1DaySelect          = 1;
    sectorPricesFiveDayAvgGradientSelect = 0;
    companyDatesPower2Select             = 1;
    companyDatesPower3Select             = 1;
    knotPointsSelect                     = 1;

    onesLr                           = 1e-07;
    datesLr                          = 1e-16;
    companyPricesMinus1DayLr         = 1e-10;
    dayNumLr                         = 1e-8;
    companyVolumesMinus1DayLr        = 1e-22;
    sectorVolumesMinus1DayLr         = 1e-21;
    companyPriceFiveDayAvgGradientLr = 1e-7;
    sectorPricesMinus1DayLr          = 1e-6;
    sectorPricesFiveDayAvgGradientLr = 1e-4;
    companyDatesPower2Lr             = 1e-16;
    companyDatesPower3Lr             = 1e-20;
    knotPointsLr                     = 1e-10;
    
    learningRates = [onesLr; datesLr; companyPricesMinus1DayLr; dayNumLr; companyVolumesMinus1DayLr; sectorVolumesMinus1DayLr;
        companyPriceFiveDayAvgGradientLr; sectorPricesMinus1DayLr; sectorPricesFiveDayAvgGradientLr; companyDatesPower2Lr;
        companyDatesPower3Lr; ones(size(knotPoints))' * knotPointsLr];
    
    featureSelection = [onesSelect; datesSelect; companyPricesMinus1DaySelect; dayNumSelect; companyVolumesMinus1DaySelect; sectorVolumesMinus1DaySelect;
        companyPriceFiveDayAvgGradientSelect; sectorPricesMinus1DaySelect; sectorPricesFiveDayAvgGradientSelect; companyDatesPower2Select;
        companyDatesPower3Select; ones(size(knotPoints))' * knotPointsSelect];
    
    learningRates = learningRates .* featureSelection;

    %% Set up company & sector data matrix

    companyData = extractDataFiles(companyDataFileName);
    sectorData = extractDataFiles(sectorDataFileName);

    %% Cut down dataset to future

    companyData = companyData(companyData(:,1) >= endDate,:);
    sectorData = sectorData(sectorData(:,1) >= endDate,:);

    %% Generate features

    testX = generateFeatures(companyData, sectorData, knotPoints, featureSelection);
    testY = companyData(:,3);
    
    %% Call linear (gradient descent)
    
    [mseGradDesc, thetasGradDesc, trainX, trainY, trainDateOffset, fittedCurve] = gradDescLinear(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints, learningRates, momentumFactor, featureSelection);
    predictedPrices = testX * thetasGradDesc;
    figure('Name','Gradient descent linear regression','NumberTitle','off');
    plotResults(companyData, trainX, trainY, trainDateOffset, testY, fittedCurve, predictedPrices)    
    
    %% Call linear (fminunc)
    
    [mseNoReg, thetasNoReg, trainX, trainY, trainDateOffset, fittedCurve] = linear(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints, featureSelection);
    predictedPrices = testX * thetasNoReg;
    figure('Name','Linear regression WITHOUT regularisation','NumberTitle','off');
    plotResults(companyData, trainX, trainY, trainDateOffset, testY, fittedCurve, predictedPrices)    
    
    %% Call reglinear 
    
    global lambda; % For return value
    [mseReg, thetasReg, trainX, trainY, trainDateOffset, fittedCurve] = reglinear(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints);
    lambd = lambda;
    predictedPrices = testX * thetasReg;
    figure('Name','Linear regression WITH regularisation','NumberTitle','off');
    plotResults(companyData, trainX, trainY, trainDateOffset, testY, fittedCurve, predictedPrices)

end