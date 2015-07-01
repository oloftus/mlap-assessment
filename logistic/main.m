function [mcrReg , thetasReg, lambd, mcrNoReg, thetasNoReg] = main()

    %% Parameters
    
    numKFolds = 5;
    companyDataFileName = 'data/company_stock_price.csv';
    sectorDataFileName = 'data/sector_stock_price.csv';
    endDate = datenum('22/10/2009', 'dd/mm/yyyy');
    knotPoints = [0, 10, 30, 80, 130];
    
    onesSelect                           = 1;
    datesSelect                          = 1;
    companyPricesMinus1DaySelect         = 1;
    dayNumSelect                         = 1;
    companyVolumesMinus1DaySelect        = 1;
    sectorVolumesMinus1DaySelect         = 1;
    companyPriceFiveDayAvgGradientSelect = 1;
    sectorPricesMinus1DaySelect          = 1;
    sectorPricesFiveDayAvgGradientSelect = 1;
    companyDatesPower2Select             = 1;
    companyDatesPower3Select             = 1;
    knotPointsSelect                     = 1;
    
    featureSelection = [onesSelect; datesSelect; companyPricesMinus1DaySelect; dayNumSelect; companyVolumesMinus1DaySelect; sectorVolumesMinus1DaySelect;
        companyPriceFiveDayAvgGradientSelect; sectorPricesMinus1DaySelect; sectorPricesFiveDayAvgGradientSelect; companyDatesPower2Select;
        companyDatesPower3Select; ones(size(knotPoints))' * knotPointsSelect];

    %% Set up company & sector data matrix

    companyData = extractDataFiles(companyDataFileName);
    sectorData = extractDataFiles(sectorDataFileName);

    %% Cut down dataset to future

    companyData = companyData(companyData(:,1) >= endDate,:);
    sectorData = sectorData(sectorData(:,1) >= endDate,:);

    %% Generate features.

    testX = generateFeatures(companyData, sectorData, knotPoints, featureSelection);
    testY = generateClasses(companyData);

    testDates = companyData(:,1);
    
    %% Call logistic without regularisation

    [mcrNoReg , thetasNoReg] = logistic(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints, featureSelection);
    classes = unique(double(testY));
    predictions = predict(testX, thetasNoReg, classes);
    figure('Name','Logistic regression WITHOUT regularisation','NumberTitle','off');
    scatter(testDates, testY, 'MarkerEdgeColor', 'black');
    hold on
    scatter(testDates, predictions, 'MarkerEdgeColor', 'green');

    %% Call logistic with regularisation
    
    global lambda;
    [mcrReg, thetasReg] = reglogistic(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints);
    lambd = lambda;
    classes = unique(double(testY));
    predictions = predict(testX, thetasReg, classes);
    figure('Name','Logistic regression WITH regularisation','NumberTitle','off');
    scatter(testDates, testY);
    hold on
    scatter(testDates, predictions);
    
end