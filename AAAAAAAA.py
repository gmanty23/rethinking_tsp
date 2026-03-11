import torch
from torch.utils.data import DataLoader
from utils.functions import load_model, move_to

# Pegamos las funciones matemáticas exactas
def compute_embedding_variance(embeddings):
    node_variance = torch.var(embeddings, dim=1, unbiased=False) 
    return node_variance.mean(dim=-1).mean()

def compute_dirichlet_energy(embeddings, graph=None):
    """
    Calcula la Energía de Dirichlet promedio del batch.
    A prueba de fallos: Si 'graph' está vacío o es inválido, asume Fully Connected.
    """
    B, N, D = embeddings.shape
    # Distancias cuadradas entre todos los pares: ||h_i - h_j||^2 -> (B, N, N)
    sq_dists = torch.cdist(embeddings, embeddings, p=2).pow(2)
    
    # Máscara por defecto: Fully Connected (todos con todos, excluyendo la diagonal)
    mask = torch.ones(B, N, N, device=embeddings.device) - torch.eye(N, device=embeddings.device).unsqueeze(0)
    
    # Si recibimos un grafo, vamos a limpiarlo y verificar si tiene aristas reales
    if graph is not None and isinstance(graph, torch.Tensor) and graph.dim() == 3 and graph.shape[1] == N:
        # 1. Binarizar por si el tensor contiene costes/pesos en lugar de adyacencia
        binary_graph = (graph != 0).float()
        
        # 2. Eliminar las conexiones del nodo consigo mismo (self-loops)
        binary_graph = binary_graph * mask
        
        # 3. Solo usamos este grafo si realmente tiene aristas definidas
        if binary_graph.sum() > 0:
            mask = binary_graph

    # Calcular la energía promedio sobre las aristas válidas de la máscara
    energy_per_graph = (sq_dists * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-9)
        
    return energy_per_graph.mean()

def check_model_energy(model_path, dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_args = load_model(model_path)
    model.to(device)
    model.eval()
    
    # Cargar solo 1 batch de 2 grafos
    dataset = model.problem.make_dataset(
        filename=dataset_path, batch_size=2, num_samples=2, 
        neighbors=train_args.get('neighbors', 20),
        knn_strat=train_args.get('knn_strat', 'None'),
        node_feature_type=train_args.get('node_feature_type', 'coords'), supervised=True
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    nodes, graph = move_to(batch['nodes'], device), move_to(batch['graph'], device)
    
    with torch.no_grad():
        h = model._init_embed(nodes)
        embeddings = model.embedder(h, graph)
        
        var = compute_embedding_variance(embeddings)
        energy = compute_dirichlet_energy(embeddings, graph)
        
        print("="*50)
        print(f"DIAGNOSTIC FOR: {model_path.split('/')[-2]}")
        print(f"Graph Tensor Shape: {graph.shape if graph is not None else 'None'}")
        print(f"Embedding Variance: {var.item():.4f}")
        print(f"Dirichlet Energy:   {energy.item():.4f}")
        print("="*50)

if __name__ == "__main__":
    # --- CAMBIA ESTAS RUTAS ---
    # Pon la ruta a uno de los modelos "0.0000" (ej. exp1_coords_ent0.01)
    MODEL_PATH = "outputs/windy_tsp_20-20/002_exp1_confidence_og_stats/05ConfOgStats_ent0.1_exp1/exp1_coords_ent0.1_20251223T021447/epoch-99.pt"
    DATASET_PATH = "data/windy_tsp/windy_tsp20_val.pkl"
    
    check_model_energy(MODEL_PATH, DATASET_PATH)