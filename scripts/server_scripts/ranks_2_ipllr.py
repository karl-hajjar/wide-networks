# columns = ['W0', 'Delta_W_1', 'Delta_W_2', 'max']
# df = pd.DataFrame(columns=columns, index=range(1, L + 2))
# df.index.name = 'layer'
#
# for l in df.index:
#     df.loc[l, columns] = [torch.matrix_rank(W0[l], tol=1e-9).item(),
#                           torch.matrix_rank(Delta_W_1[l], tol=1e-9).item(),
#                           torch.matrix_rank(Delta_W_2[l], tol=1e-9).item(),
#                           min(W0[l].shape[0], W0[l].shape[1])]
#
# df.loc[:, 'batch_size'] = batch_size
# df
#
# # ## Explore at step 1
#
# # ### On all training examples
#
# x, y = full_x, full_y
#
# with torch.no_grad():
#     x1 = {0: x}
#     h0 = {1: F.linear(x, W0[1], b0)}
#     delta_h_1 = {1: F.linear(x, Delta_W_1[1], Delta_b_1)}
#     h1 = {1: layer_scales[0] * ipllr_1.input_layer.forward(x) / math.sqrt(ipllr_1.d + 1)}
#     x1[1] = ipllr_1.activation(h1[1])
#
# torch.testing.assert_allclose(h0[1] + delta_h_1[1], h1[1], rtol=1e-5, atol=1e-5)
#
# with torch.no_grad():
#     for i, l in enumerate(range(2, L + 1)):
#         layer_1 = getattr(ipllr_1.intermediate_layers, intermediate_layer_keys[i])
#         x = x1[l - 1]
#
#         h0[l] = F.linear(x, W0[l])
#         delta_h_1[l] = F.linear(x, Delta_W_1[l])
#
#         h1[l] = layer_scales[l - 1] * layer_1.forward(x)
#         x1[l] = ipllr_1.activation(h1[l])
#
#         torch.testing.assert_allclose(h0[l] + delta_h_1[l], h1[l], rtol=1e-5, atol=1e-5)
#
# with torch.no_grad():
#     x = x1[L]
#     h0[L + 1] = F.linear(x, W0[L + 1])
#     delta_h_1[L + 1] = F.linear(x, Delta_W_1[L + 1])
#     h1[L + 1] = layer_scales[L] * ipllr_1.output_layer.forward(x)
#     x1[L + 1] = ipllr_1.activation(h1[L + 1])
#
#     torch.testing.assert_allclose(h0[L + 1] + delta_h_1[L + 1], h1[L + 1], rtol=1e-5, atol=1e-5)
#
# # ##### Diversity
#
# columns = ['h0', 'delta_h_1', 'h1']
# df = pd.DataFrame(columns=columns, index=range(1, L + 2))
# df.index.name = 'layer'
# bs = x.shape[0]
#
# for l in df.index:
#     maxes = dict()
#
#     _, maxes['h0'] = torch.max(h0[l], dim=1)
#     _, maxes['delta_h_1'] = torch.max(delta_h_1[l], dim=1)
#     _, maxes['h1'] = torch.max(h1[l], dim=1)
#
#     df.loc[l, columns] = [maxes[key].unique().numel() for key in columns]
#     df.loc[l, 'max'] = min(bs, h0[l].shape[1])
#
# df.loc[:, 'batch_size'] = bs
# df.loc[:, 'max'] = df.loc[:, 'max'].astype(int)
# df
#
# # In[91]:
#
# columns = ['h0', 'delta_h_1', 'h1', 'x1']
# df = pd.DataFrame(columns=columns, index=range(1, L + 2))
# df.index.name = 'layer'
# bs = x.shape[0]
#
# for l in df.index:
#     df.loc[l, columns] = [torch.matrix_rank(h0[l], tol=1e-8).item(),
#                           torch.matrix_rank(delta_h_1[l], tol=1e-8).item(),
#                           torch.matrix_rank(h1[l], tol=1e-8).item(),
#                           torch.matrix_rank(x1[l], tol=1e-8).item()]
#     df.loc[l, 'max'] = min(h0[l].shape[0], h0[l].shape[1])
#
# df.loc[:, 'n_el'] = bs
# df.loc[:, 'max'] = df.loc[:, 'max'].astype(int)
# df