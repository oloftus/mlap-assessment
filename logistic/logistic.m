function [mcr, thetas] = logistic(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints, featureSelection)
   
    %% Set up company & sector data matrix
    
    companyData = extractDataFiles(companyDataFileName);
    sectorData = extractDataFiles(sectorDataFileName);
    
    %% Cut down dataset to the previous year
    
    oneYearAgo = addtodate(endDate, -1, 'year');
    companyData = companyData(companyData(:,1) < endDate & companyData(:,1) > oneYearAgo,:);
    sectorData = sectorData(sectorData(:,1) < endDate & sectorData(:,1) > oneYearAgo,:);
    
    %% Generate features & classes
    
    trainX = generateFeatures(companyData, sectorData, knotPoints, featureSelection);
    trainY = generateClasses(companyData);

    %% K-fold cross validation
    
    mcr = crossvalCustomMcr(trainX, trainY, @solveFminuncWrapper, numKFolds);
    
    %% Retrain to obtain model parameters
    
    thetas = solveFminunc(trainX, trainY);
    
end
