import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. FUNCIONES DE GENERACIÓN Y CÁLCULO DE COSTES
# ==========================================

def generate_windy_instance(num_nodes, alpha=1.0, max_wind=0.5):
    loc = np.random.rand(num_nodes, 2)
    wind_angle = np.random.uniform(0, 2 * np.pi)
    wind_mag = np.random.uniform(0, max_wind)
    wind = np.array([wind_mag * np.cos(wind_angle), wind_mag * np.sin(wind_angle)])
    return {'loc': loc, 'wind': wind, 'alpha': alpha}

def calculate_cost_matrix(loc, wind, alpha):
    n = len(loc)
    diff = loc[None, :, :] - loc[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    dists[np.arange(n), np.arange(n)] = 1.0 
    u = diff / dists[:, :, None]
    u[np.arange(n), np.arange(n)] = 0 
    wind_proj = np.dot(u, wind)
    
    # FÓRMULA EXPONENCIAL: C = d * exp(-alpha * proj)
    # Proj > 0 (A favor) -> exp(-x) < 1 -> Coste baja
    # Proj < 0 (En contra) -> exp(+x) > 1 -> Coste sube
    costs = dists * np.exp(-1.0 * alpha * wind_proj)
    costs[np.arange(n), np.arange(n)] = 0
    return costs

# ==========================================
# 2. SCRIPT DE VISUALIZACIÓN (6 Gráficos)
# ==========================================

def visualize_windy_tsp():
    # Configuración de parámetros para la demo
    N = 50
    ALPHA = 2.5      # Valor alto para que la curva se note bien
    MAX_WIND = 0.6   # Viento fuerte
    
    # Generar instancia
    data = generate_windy_instance(N, alpha=ALPHA, max_wind=MAX_WIND)
    coords = data['loc']
    wind = data['wind']
    alpha = data['alpha']
    
    # Calcular matrices
    costs = calculate_cost_matrix(coords, wind, alpha)
    
    # Recalcular datos auxiliares para los gráficos
    diff = coords[None, :, :] - coords[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Evitar div por 0 en diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        u = diff / dists[:, :, None]
    wind_proj = np.dot(u, wind) # Matriz de proyecciones
    
    # Preparar figura
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # -----------------------------------------------------------
    # GRÁFICO 1: Mapa Físico (Topología)
    # -----------------------------------------------------------
    ax = axs[0, 0]
    ax.scatter(coords[:,0], coords[:,1], c='blue', edgecolors='k', s=50, alpha=0.7, label='Nodos')
    # Dibujar flecha de viento
    center = [0.5, 0.5]
    scale_factor = 0.2
    ax.arrow(center[0], center[1], wind[0]*scale_factor, wind[1]*scale_factor, 
             head_width=0.04, fc='red', ec='red', width=0.005)
    ax.text(center[0] + wind[0]*scale_factor, center[1] + wind[1]*scale_factor, 
            f' Viento\n(|w|={np.linalg.norm(wind):.2f})', color='red', fontsize=10, fontweight='bold')
    ax.set_title("1. Mapa de Ciudades y Viento", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    # -----------------------------------------------------------
    # GRÁFICO 2: La Curva Exponencial (Validación Física) 

#[Image of Exponential decay graph]

    # -----------------------------------------------------------
    ax = axs[0, 1]
    # Aplanamos matrices excluyendo diagonal
    mask = ~np.eye(N, dtype=bool)
    x_vals = wind_proj[mask]       # Proyección (u . w)
    y_vals = costs[mask] / dists[mask] # Factor de coste (Coste / Distancia Euclidea)
    
    ax.scatter(x_vals, y_vals, alpha=0.2, color='purple', s=10, label='Aristas Generadas')
    
    # Línea Teórica: y = exp(-alpha * x)
    x_theory = np.linspace(min(x_vals), max(x_vals), 100)
    y_theory = np.exp(-alpha * x_theory)
    ax.plot(x_theory, y_theory, 'r-', linewidth=2, label=r'Teórico: $e^{-\alpha (u \cdot w)}$')
    
    ax.set_title(f"2. Validación de Fórmula (Alpha={alpha})", fontsize=12)
    ax.set_xlabel("Alineación con el Viento (Proyección)")
    ax.set_ylabel("Factor de Multiplicación del Coste")
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.text(min(x_vals), max(y_vals)*0.9, "Contra Viento\n(Coste > 1.0)", color='darkred')
    ax.text(max(x_vals)*0.5, min(y_vals), "A Favor\n(Coste < 1.0)", color='green')
    ax.legend()

    # -----------------------------------------------------------
    # GRÁFICO 3: Impacto en Distancias (Euclídea vs Windy)
    # -----------------------------------------------------------
    ax = axs[0, 2]
    d_flat = dists[mask]
    c_flat = costs[mask]
    
    ax.scatter(d_flat, c_flat, alpha=0.3, c=x_vals, cmap='coolwarm_r', s=10)
    # Línea de referencia y=x (sin viento)
    max_val = max(d_flat.max(), c_flat.max())
    ax.plot([0, max_val], [0, max_val], 'k--', label='Sin Viento (y=x)')
    
    ax.set_title("3. Distancia Euclidiana vs Coste Windy", fontsize=12)
    ax.set_xlabel("Distancia Real (Euclidiana)")
    ax.set_ylabel("Coste Final")
    ax.text(max_val*0.1, max_val*0.8, "Rojo: Contra Viento", color='red')
    ax.text(max_val*0.6, max_val*0.2, "Azul: A Favor", color='blue')

    # -----------------------------------------------------------
    # GRÁFICO 4: Histograma de Asimetría
    # -----------------------------------------------------------
    ax = axs[1, 0]
    # Calculamos diferencia relativa entre ir y volver
    asymmetry = []
    for i in range(N):
        for j in range(i+1, N):
            c_ij = costs[i, j]
            c_ji = costs[j, i]
            # Ratio: cuantas veces es más caro el camino malo vs el bueno
            ratio = max(c_ij, c_ji) / min(c_ij, c_ji)
            asymmetry.append(ratio)
            
    sns.histplot(asymmetry, kde=True, ax=ax, color='orange', bins=30)
    ax.set_title("4. Grado de Asimetría (Ratio Ida/Vuelta)", fontsize=12)
    ax.set_xlabel("Ratio Coste (Peor Dirección / Mejor Dirección)")
    ax.axvline(1.0, color='k', linestyle='--')
    ax.text(1.05, ax.get_ylim()[1]*0.9, "1.0 = Simétrico", rotation=90)

    # -----------------------------------------------------------
    # GRÁFICO 5: Matriz de Costes (Heatmap)
    # -----------------------------------------------------------
    ax = axs[1, 1]
    sns.heatmap(costs, ax=ax, cmap="viridis", cbar_kws={'label': 'Coste'})
    ax.set_title("5. Matriz de Costes Completa", fontsize=12)
    ax.set_xlabel("Destino")
    ax.set_ylabel("Origen")

    # -----------------------------------------------------------
    # GRÁFICO 6: Reloj de Costes (Polar / Angular)
    # -----------------------------------------------------------
    ax = axs[1, 2]
    # Calcular ángulo de cada arista respecto al eje X
    dx = diff[:, :, 0][mask]
    dy = diff[:, :, 1][mask]
    edge_angles = np.arctan2(dy, dx)
    
    # Calcular ángulo del viento
    wind_angle_rad = np.arctan2(wind[1], wind[0])
    
    # Diferencia angular relativa al viento
    rel_angles = edge_angles - wind_angle_rad
    # Normalizar a [-pi, pi]
    rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi
    
    factors = costs[mask] / dists[mask]
    
    ax.scatter(rel_angles, factors, s=10, alpha=0.4, c=factors, cmap='viridis')
    ax.set_title("6. Coste vs Ángulo Relativo al Viento", fontsize=12)
    ax.set_xlabel("Ángulo relativo (0 = Viento de cola)")
    ax.set_ylabel("Factor de Coste")
    ax.set_xlim(-np.pi, np.pi)
    # Marcar puntos clave
    ax.axvline(0, color='green', linestyle='--', label='Viento a favor')
    ax.axvline(-np.pi, color='red', linestyle='--', label='Viento en contra')
    ax.axvline(np.pi, color='red', linestyle='--')
    ax.legend(loc='upper right', fontsize='small')

    # Finalizar
    plt.suptitle(f"Análisis de Generación Windy TSP (Exponencial)\nN={N}, Alpha={alpha}, |W|={np.linalg.norm(wind):.3f}", fontsize=16)
    plt.show()

if __name__ == "__main__":
    visualize_windy_tsp()