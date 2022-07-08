function [loss,loss_partial] = obj_function_eta_dop_zenith_partial(x,dop,zenith)
eta=x;
[predict_dop,predict_dop_partial] = compute_d_dop_partial(zenith,eta');

dop_diff = predict_dop-dop;
loss = mean(dop_diff.^2,'all');
loss_partial = 2.*(mean(dop_diff.*predict_dop_partial,'all'));

end