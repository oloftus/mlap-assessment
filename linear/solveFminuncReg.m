function thetas = solveFminuncReg(xTrain, yTrain)

    global lambda;
    
    function [loss, grad] = squaredLoss(thetas)
        errors = yTrain - xTrain * thetas;
        sqLoss = errors' * errors;
        
%         Lasso
        loss = lambda * sqLoss + (1 - lambda) * sum(abs(thetas));
        grad = -2 * lambda * xTrain' * errors + (1 - lambda) * sign(thetas);

%         Ridge
%         loss = lambda * sqLoss + (1 - lambda) * sum(abs(thetas) .^ 2);
%         grad = -2 * lambda * xTrain' * errors + 2 * (1 - lambda) * thetas;
    end

    thetas = zeros(size(xTrain, 2), 1);
    opts = optimset('GradObj','on');
    thetas = fminunc(@squaredLoss, thetas, opts);
    
end
