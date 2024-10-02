classdef finalFunctions
   methods
       function plot_time_series(~, time_series)
            % Plot time series
            plot(time_series.Time, time_series.Data);
            hold on;
            
            % Fit trend line using linear regression
            X = (1:length(time_series.Data)).'; % Create a column vector of indices
            fit = fitlm(X, time_series.Data); % Fit linear model
            trend_line = fit.Coefficients.Estimate(1) + fit.Coefficients.Estimate(2) * X; % Compute trend line
            
            % Plot trend line
            plot(time_series.Time, trend_line, 'r', 'LineWidth', 1.5);
            
            % Plot mean line
            mean_value = mean(time_series.Data);
            mean_line = ones(size(time_series.Data)) * mean_value;
            plot(time_series.Time, mean_line, 'b', 'LineWidth', 1.5);
            
            % Add legend and labels
            legend('Time Series', 'Trend Line', 'Mean Line');
            xlabel('Time');
            ylabel('Value');
            
            hold off;
       end
        
        function plot_histogram(~, time_series)
            % Extract data from the time series object
            data = time_series.Data;
            
            % Plot the histogram
            histogram(data, 'FaceColor', 'blue');
            
            % Add labels and title
            xlabel('Value');
            ylabel('Frequency');
        end
        
        function plot_acf(~, data)
            % Calculate autocorrelation function
            acf_values = autocorr(data, 30);
        
            % Calculate number of observations
            n = length(data);
        
            % Calculate standard error for each ACF value
            se = 1 / sqrt(n);
        
            % Calculate critical value for 95% confidence interval
            z = norminv(1 - 0.05 / 2);
        
            % Calculate confidence interval bounds
            ci_bounds = z * se * ones(size(acf_values));
        
            % Plot the ACF with confidence intervals
            stem(0:30, acf_values);
            hold on;
            plot(0:30, ci_bounds, 'r--', 'LineWidth', 2);
            plot(0:30, -ci_bounds, 'r--', 'LineWidth', 2);
            hold off;
        
            % Add labels and title
            xlabel('Lag');
            ylabel('Autocorrelation');
            title('ACF with 95% Confidence Interval');
        end


        function plot_pacf(~, data)
            % Calculate partial autocorrelation function
            pacf_values = parcorr(data, 30);
            
            % Calculate standard error for each PACF value
            n = length(data);
            se = 1 / sqrt(n);
            
            % Calculate critical value for 95% confidence interval
            z = norminv(1 - 0.05 / 2);
            
            % Calculate confidence interval bounds
            ci_bounds = z * se * ones(size(pacf_values));
            
            % Plot the PACF with confidence intervals
            stem(0:30, pacf_values, 'filled', 'MarkerSize', 4);
            hold on;
            plot(0:30, ci_bounds, 'r--', 'LineWidth', 2);
            plot(0:30, -ci_bounds, 'r--', 'LineWidth', 2);
            hold off;
            
            % Add labels and title
            xlabel('Lag');
            ylabel('Partial Autocorrelation');
            title('PACF with 95% Confidence Interval');
        end
        
        function [EstMdl, AIC] = fit_sarima_model(~, train_data, p, D, q)
            % Create SARIMA model
            Mdl = arima(p,D,q);
        
            % Fit the SARIMA model
            EstMdl = estimate(Mdl,train_data);
        
            results = summarize(EstMdl);
            AIC = results.AIC;
        
            % Display AIC
            disp(['AIC of the fitted model: ', num2str(AIC)]);
        end
   end
end
        
      

