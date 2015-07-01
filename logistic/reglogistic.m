function [mcr, thetas] = reglogistic(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints)

    global lambda;
    
    featureSelection = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; ones(size(knotPoints))' * 1];
    
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

    %% Find a suitable lambda value
    
    numLambdas = 10;
    
    mcrs = zeros(1, numLambdas);
    lambdas = linspace(0.1, 1, numLambdas);
    
    for ix = 1:numLambdas
        lambda = lambdas(ix); %#ok<NASGU>
        mcrs(ix) = crossvalCustomMcr(trainX, trainY, @solveFminuncRegWrapper, numKFolds);
    end
    
    lambda = lambdas(mcrs == min(mcrs));
    
    %% K-fold cross validation
    
    mcr = crossvalCustomMcr(trainX, trainY, @solveFminuncRegWrapper, numKFolds);
    
    %% Retrain to obtain model parameters
    
    thetas = solveFminuncReg(trainX, trainY);
    
end
