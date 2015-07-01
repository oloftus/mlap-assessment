function [mse, thetas, trainX, trainY, trainDateOffset, fittedCurve] = linear(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints, featureSelection)

    %% Set up company & sector data matrix
    
    companyData = extractDataFiles(companyDataFileName);
    sectorData = extractDataFiles(sectorDataFileName);
    
    %% Cut down dataset to the previous year
    
    oneYearAgo = addtodate(endDate, -1, 'year');
    companyData = companyData(companyData(:,1) < endDate & companyData(:,1) > oneYearAgo,:);
    sectorData = sectorData(sectorData(:,1) < endDate & sectorData(:,1) > oneYearAgo,:);
    
    %% Generate features
    
    trainDateOffset = companyData(1,1);
    trainX = generateFeatures(companyData, sectorData, knotPoints, featureSelection);
    trainY = companyData(:,3);

    %% K-fold cross validation

    mse = crossvalCustomMse(trainX, trainY, @solveFminuncWrapper, numKFolds);
    
    %% Retrain to obtain model parameters
    
    thetas = solveFminunc(trainX, trainY);
    fittedCurve = trainX * thetas;
    
end
