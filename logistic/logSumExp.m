function out = logSumExp(xis)

    maxXi = max(xis);
    out = maxXi + log(sum(exp(xis - maxXi)));
end
