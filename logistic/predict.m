function prediction = predict(inputs, thetas, classes)

    probs = exp(inputs * thetas) ./ repmat(sum(exp(inputs * thetas), 2), 1, size(classes, 1));
    prediction = (probs == repmat(max(probs, [], 2), 1, size(classes, 1))) * classes;
end
