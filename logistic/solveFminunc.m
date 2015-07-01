function thetas = solveFminunc(trainX, trainY)

    [numDataRows, numFeatures] = size(trainX);
    numClasses = size(unique(trainY), 1);
    
    function [loss, grad] = lossFn(thetas)
        logProb = 0;
        grad = zeros(numFeatures, numClasses);

        for row = 1:numDataRows
            x = trainX(row, :);
            y = trainY(row);
            
            logProb = logProb + x * thetas(:, y + 1) - logSumExp(x * thetas);
            
            indicator = zeros(1, numClasses);
            indicator(1, y + 1) = 1;
            sumExp = sum(exp(x * thetas));

            grad = grad - x' * (indicator - exp(x * thetas) / sumExp);
        end
        
        loss = -logProb;
    end
    
    thetas = zeros(numFeatures, numClasses);
    opts = optimset('GradObj','on','MaxFunEvals',100,'MaxIter',100);
    thetas = fminunc(@lossFn, thetas, opts);
end
