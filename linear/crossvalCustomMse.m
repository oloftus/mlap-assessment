function mse = crossvalCustomMse(trainX, trainY, solveWrapper, numKFolds)

    permittedSplitRatioVariance = 0.1;
    
    while 1
        foldAllocations = ceil(rand(size(trainX, 1), 1) * numKFolds);
        folds = unique(foldAllocations)';
        numDataRows = size(trainX, 1);
        counts = sum(repmat(foldAllocations, 1, numKFolds) == repmat(folds, numDataRows, 1));
        splitRatio = 1 / numKFolds;
        
        partitionedEqually = all(abs(counts ./ sum(counts) - splitRatio) < permittedSplitRatioVariance * splitRatio);
        
        if partitionedEqually
            break;
        end
    end

    partTrainX = [foldAllocations, trainX]; % First row contains fold ID
    partTrainY = [foldAllocations, trainY];
    
    mses = zeros(numKFolds, 1);
    
    for fold = folds
        foldIds = partTrainX(:,1);

        trainXFold = partTrainX(foldIds ~= fold, 2:end);
        trainYFold = partTrainY(foldIds ~= fold, 2:end);
        testX = partTrainX(foldIds == fold, 2:end);
        testY = partTrainY(foldIds == fold, 2:end);
        
        yfit = solveWrapper(trainXFold, trainYFold, testX);
        
        error = yfit - testY;
        sqError = error .^2 ;
        mses(fold) = mean(sqError);
    end

    mse = mean(mses);
end
