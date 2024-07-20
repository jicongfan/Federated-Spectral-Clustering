function [X, y] = generateConcentricCircles(K, sigma, segDistCircle)
% content: randomly generate some concentric circles
%
theta = 0:.01:2*pi;
theta = theta';
ThetaLen = length(theta);
X = zeros(K*ThetaLen, 2);
y = zeros(K*ThetaLen, 1);
en = ones(ThetaLen, 1);
map = rand(K, 3);
figure;
for i = 1: K
    r = segDistCircle*i;
    x1 = r*cos(theta) + sigma*randn(ThetaLen, 1);
    x2 = r*sin(theta) + sigma*randn(ThetaLen, 1);
    X((i - 1)*ThetaLen + 1: i*ThetaLen, :) = [x1, x2];
    y((i - 1)*ThetaLen + 1: i*ThetaLen, :) = i*en;
    plot(x1, x2, '.');
    colormap(map(i, :));
    axis equal, hold on;
end

plot(0, 0, 'mo', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'm', 'markersize', 8);
xlabel('x_1');
ylabel('x_2');
hold off;

end