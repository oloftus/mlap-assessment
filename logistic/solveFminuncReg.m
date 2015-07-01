function thetas = solveFminuncReg(trainX, trainY)

    global lambda;

    [numDataRows, numFeatures] = size(trainX);
    numClasses = size(unique(trainY), 1);
    
    function [loss, grads] = lossFn(thetas)
        logProb = 0;
        grads = zeros(numFeatures, numClasses);

        for row = 1:numDataRows
            x = trainX(row, :);
            y = trainY(row);
            
            logProb = logProb + lambda * x * thetas(:, y + 1) - logSumExp(x * thetas);
            
            indicator = zeros(1, numClasses);
            indicator(1, y + 1) = 1;
            sumExp = sum(exp(x * thetas));

            grads = grads - lambda * x' * (indicator - exp(x * thetas) / sumExp);
        end
        
        logProb = logProb + (1 - lambda) * sum(sum(abs(thetas)));
        grads = grads + (1 - lambda) * sign(thetas);
        loss = -logProb;
    end
    
    thetas = zeros(numFeatures, numClasses);
    opts = optimset('GradObj','on','MaxFunEvals',100,'MaxIter',100);
    thetas = fminunc(@lossFn, thetas, opts);
end
