# Propuesta de Nuevos Métodos para `InteractomeAnalyzer`

A continuación se detallan varios métodos genéricos propuestos para extender la funcionalidad de la clase `InteractomeAnalyzer` en `interactome.py`. Estos métodos están diseñados para facilitar el análisis estadístico, la visualización de redes y la comparación de resultados entre diferentes motores de predicción.

## 1. Filtrado Avanzado y Selección

### `filter_by_metrics(self, criteria: Dict[str, Tuple[float, float]]) -> pd.DataFrame`
Permite filtrar el interactoma completo basándose en múltiples rangos de métricas simultáneamente.
- **Entrada:** Un diccionario donde las llaves son nombres de columnas y los valores son tuplas `(min, max)`.
- **Utilidad:** Facilita la extracción rápida de subconjuntos de datos (ej. "Alta confianza y baja profundidad de MSA").

### `get_top_interactions(self, metric: str = "ipSAE", top_n: int = 10, ascending: bool = False) -> pd.DataFrame`
Devuelve las mejores $N$ interacciones según una métrica específica.
- **Utilidad:** Identificación rápida de los candidatos más prometedores para validación experimental.

## 2. Análisis a Nivel de Proteína

### `summarize_by_protein(self) -> pd.DataFrame`
Genera un resumen estadístico para cada proteína individual presente en el interactoma.
- **Métricas calculadas:** Número de socios (degree), promedio de ipSAE, promedio de pDockQ2, mejor socio detectado.
- **Utilidad:** Identificar "hubs" en el interactoma o proteínas que tienden a formar complejos muy estables.

### `detect_residue_hotspots(self, min_confidence: float = 0.5) -> pd.DataFrame`
Identifica residuos que participan frecuentemente en interfaces de alta confianza a través de múltiples socios.
- **Utilidad:** Mapeo de sitios de unión "multitarea" o dominios de interacción clave.

## 3. Integración y Redes

### `export_to_network(self, output_format: str = "cytoscape") -> pd.DataFrame`
Exporta los datos en un formato de lista de aristas (Edge List) compatible con herramientas como Cytoscape o Gephi.
- **Atributos de arista:** ipSAE, pDockQ2, Tier.
- **Utilidad:** Visualización de la topología del interactoma viral o virus-huésped.

## 4. Comparación y Validación

### `compare_engines(self, other_interactome_df: pd.DataFrame, suffix_a: str = "_af3", suffix_b: str = "_boltz") -> pd.DataFrame`
Compara métricas entre dos ejecuciones del mismo interactoma realizadas con diferentes motores (ej. AF3 vs Boltz).
- **Utilidad:** Evaluar la consistencia de las predicciones y reducir falsos positivos mediante consenso.

### `calculate_orthology_consistency(self, orthology_map: Dict[str, str]) -> pd.DataFrame`
Si se dispone de un mapa de ortólogos, verifica si las interacciones predichas en una especie se mantienen en otra.
- **Utilidad:** Análisis evolutivo de la conservación de complejos proteicos.

## 5. Análisis de Grupos (Clustering)

### `cluster_interactome_by_metrics(self, n_clusters: int = 4) -> pd.DataFrame`
Aplica algoritmos de aprendizaje no supervisado (ej. K-means) para agrupar las interacciones basándose en el vector de métricas estructurales (pLDDT, ipSAE, pDockQ2, etc.).
- **Utilidad:** Descubrir patrones de interacción que no son evidentes mediante umbrales simples de una sola métrica.
