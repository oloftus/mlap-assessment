function [thetas, fittedCurve] = solveGradDesc(xTrain, yTrain, momentumFactor, learningRates)

    [numDataRows, inputVectorSize] = size(xTrain);
    
    thetas = zeros(inputVectorSize, 1);
    gradients = zeros(inputVectorSize, 1);
    
    smallestLoss = Inf;
    
    while 1
        thetas = thetas + learningRates .* gradients;
        
        fittedCurve = zeros(numDataRows, 1);
        sqLoss = 0;
        
        for row = 1:numDataRows
            x = xTrain(row, :)';
            y = yTrain(row, 1);
            
            loss = y - dot(thetas, x);
            sqLoss = sqLoss + loss ^ 2;
            
            gradients = gradients + 2 * loss * x;

            fittedCurve(row, 1) = sum(x .* thetas);
        end
        
        if (sqLoss > momentumFactor * smallestLoss)
            break;
        end
        smallestLoss = sqLoss;
    end
end
