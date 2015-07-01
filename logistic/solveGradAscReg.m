function thetas = solveGradAscReg(trainX, trainY, learningRate, permittedIterations)

    global lambda;

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
            
            logProb = logProb + lambda * x * thetas(:, y) - logSumExp(x * thetas);
        
            indicator = zeros(1, numClasses);
            indicator(1, trainY(row)) = 1;
            sumExp = sum(exp(x * thetas));
            
            grads = grads + lambda * x' * (indicator - exp(x * thetas) / sumExp);
        end
        
        logProb = logProb + (1 - lambda) * sum(sum(abs(thetas)));
        grads = grads + (1 - lambda) * sign(thetas);
        
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
