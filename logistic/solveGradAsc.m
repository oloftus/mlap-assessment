function thetas = solveGradAsc(trainX, trainY, learningRate, permittedIterations)

    [numDataRows, numFeatures] = size(trainX);
    numClasses = size(unique(trainY), 1);
    
    thetas = zeros(numFeatures, numClasses);
    grads = zeros(numFeatures, numClasses);
    
    maxLogProb = 0;
    iterationsSpare = permittedIterations;
    
    while 1
        logProb = 0;
        
        for row = 1:numDataRows
            x = trainX(row, :);
            y = double(trainY(row));
            
            logProb = logProb + x * thetas(:, y) - logSumExp(x * thetas);
            
            indicator = zeros(1, numClasses);
            indicator(1, trainY(row)) = 1;
            sumExp = sum(exp(x * thetas));
            
            grads = grads + x' * (indicator - exp(x * thetas) / sumExp);
        end
        
        thetas = thetas + learningRate .* grads;
        
        if maxLogProb == 0
            maxLogProb = logProb;
        end
        
        if logProb > maxLogProb
            maxLogProb = logProb;
            iterationsSpare = permittedIterations;
        end
        
        iterationsSpare = iterationsSpare - 1;
        
        if iterationsSpare == 0
            break;
        end
    end
    
end
