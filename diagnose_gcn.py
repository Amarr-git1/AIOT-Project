import torch
import numpy as np
from data.data_processor import NetworkDataProcessor
from models.gcn_model import GCN

GCN_CHECKPOINT = 'models/gcn_pretrained.pt'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc = NetworkDataProcessor('combine.csv', num_nodes=50)
    df = proc.load_data(nrows=2000)
    if df is None:
        print('No dataset available')
        return

    graphs, labels = proc.create_synthetic_graphs(df, num_graphs=120)

    model = GCN(input_dim=8, hidden_dim=32).to(device)
    if not torch.cuda.is_available():
        state = torch.load(GCN_CHECKPOINT, map_location='cpu')
    else:
        state = torch.load(GCN_CHECKPOINT)
    model.load_state_dict(state)
    model.eval()

    # Split into validation / test
    # Split into validation / test (60/60)
    val_graphs = graphs[:60]
    val_labels = labels[:60]
    test_graphs = graphs[60:120]
    test_labels = labels[60:120]

    def avg_risks_from_graphs(glist):
        res = []
        with torch.no_grad():
            for (x, edge_index, _) in glist:
                x = x.to(device)
                edge_index = edge_index.to(device)
                out = model(x, edge_index).cpu().numpy().squeeze()
                res.append(out.mean())
        return np.array(res)

    val_avg = avg_risks_from_graphs(val_graphs)
    test_avg = avg_risks_from_graphs(test_graphs)

    # Sweep thresholds on validation to maximize F1
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    best_thr = 0.5
    best_f1 = -1
    for thr in np.linspace(0.0, 1.0, 101):
        preds = (val_avg > thr).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f'Validation best threshold={best_thr:.3f} with F1={best_f1:.4f}')

    # Evaluate on test
    test_preds = (test_avg > best_thr).astype(int)
    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)

    print('\nTest set results using threshold from validation:')
    print(f'  Accuracy={acc*100:.2f}%  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1:.4f}')
    print('\nSample test graph risks (first 10):')
    for i, val in enumerate(test_avg[:10]):
        print(f'  Graph {i}: avg_risk={val:.4f} | label={test_labels[i]} | pred={(1 if val>best_thr else 0)}')

if __name__ == "__main__":
    main()
