function classes = generateClasses(companyData)

    prices = companyData(:,3);
    pricesMinus1Day = [prices(1,1); prices(1:end - 1,1)];
    absChange = prices - pricesMinus1Day;
    perChange = absChange ./ pricesMinus1Day;
    
    noChangeClass = abs(perChange) <= 0.02;
    upClass = 0.02 < perChange & perChange <= 0.05;
    sharpUpClass = perChange > 0.05;
    downClass = -0.05 <= perChange & perChange < -0.02;
    sharpDownClass = perChange < -0.05;
    
    classes = ...
        0 * noChangeClass + ...
        1 * upClass + ...
        2 * downClass + ...
        3 * sharpUpClass + ...
        4 * sharpDownClass;

end
