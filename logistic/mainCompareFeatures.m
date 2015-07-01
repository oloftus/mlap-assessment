function mcrs = mainCompareFeatures()
    
    numKFolds = 5;
    companyDataFileName = 'data/company_stock_price.csv';
    sectorDataFileName = 'data/sector_stock_price.csv';
    endDate = datenum('22/10/2009', 'dd/mm/yyyy');
    knotPoints = [0, 10, 30, 80, 130];
    
    featureSelection = ...
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ones(size(knotPoints)) * 0;
        1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, ones(size(knotPoints)) * 0;
        1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, ones(size(knotPoints)) * 0;
        1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, ones(size(knotPoints)) * 1;
        1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, ones(size(knotPoints)) * 1];

    mcrs = zeros(size(featureSelection, 1), 1);
    for i = 1:size(featureSelection, 1);
        [mcrNoReg , ~] = logistic(numKFolds, companyDataFileName, sectorDataFileName, endDate, knotPoints, featureSelection(i, :)');
        mcrs(i) = mcrNoReg;
    end

end