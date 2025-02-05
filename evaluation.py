# Evaluation
model.eval()
y_preds, y_trues = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = model(batch_X, mask=None)
        y_preds.append(predictions.cpu().numpy())
        y_trues.append(batch_y.cpu().numpy())

y_preds = np.concatenate(y_preds, axis=0)
y_trues = np.concatenate(y_trues, axis=0)

mse = np.mean((y_preds - y_trues) ** 2)
r2 = r2_score(y_trues, y_preds)
print(f"Test MSE: {mse:.6f}, R2 Score: {r2:.6f}")

