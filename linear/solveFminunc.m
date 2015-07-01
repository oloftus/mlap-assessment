function thetas = solveFminunc(xTrain, yTrain)

    function [sqLoss, grad] = squaredLoss(thetas)
        errors = yTrain - xTrain * thetas;
        sqLoss = errors' * errors;
        grad = -2 * xTrain' * errors;
    end

    thetas = zeros(size(xTrain, 2), 1);
    opts = optimset('GradObj','on');
    thetas = fminunc(@squaredLoss, thetas, opts);
    
end
