# Evaluación de LLMs Open-Source para Sistemas de Recomendación Conversacional

Proyecto de investigación para el curso IIC3633 - Sistemas Recomendadores (2025-2)

**Integrantes**: Martín Muñoz, Eduardo Soto, Randall Biermann

---

## 📋 Descripción del Proyecto

Este proyecto evalúa y compara el rendimiento de Large Language Models (LLMs) open-source de última generación en tareas de recomendación conversacional, utilizando el dataset ReDial. Se analizan diferentes arquitecturas de modelos, tamaños de parámetros y estrategias de prompting para determinar su efectividad en generar recomendaciones personalizadas de películas basadas en conversaciones en lenguaje natural.

---

## 🗂️ Estructura del Repositorio

### Archivos Principales

#### `redial_dataset.py`
**Propósito**: Carga y preprocesa el dataset ReDial.

**Funcionalidades clave**:
- Carga conversaciones desde archivos JSONL (train/validation/test)
- Divide conversaciones en contexto y ground truth según un ratio configurable
- Formatea conversaciones en texto legible para LLMs (User/Recommender)
- Extrae películas recomendadas como ground truth
- Filtra muestras válidas para evaluación

**Uso**:
```python
from redial_dataset import ReDiALDataset

dataset = ReDiALDataset('.', split='test')
samples = dataset.get_evaluation_samples(n_samples=100, context_ratio=0.7)
```

---

#### `llm_recommender.py`
**Propósito**: Wrapper genérico para cargar y usar cualquier LLM de HuggingFace.

**Funcionalidades clave**:
- Carga modelos de HuggingFace Transformers
- Soporta cuantización 8-bit para modelos grandes
- Genera recomendaciones dado un contexto conversacional
- Extrae títulos de películas de las respuestas del modelo
- Gestión de memoria GPU

**Uso**:
```python
from llm_recommender import LLMRecommender

llm = LLMRecommender("google/gemma-2-2b-it", load_in_8bit=False)
response, latency = llm.generate_recommendations(context, prompt_fn)
movies = llm.extract_movie_titles(response)
```

---

#### `prompts.py`
**Propósito**: Define diferentes estrategias de prompting para los LLMs.

**Estrategias implementadas**:
- **Zero-shot**: Prompt básico sin ejemplos
- **Few-shot**: Prompt con ejemplos de conversaciones y recomendaciones
- **Chain-of-thought**: Prompt que pide razonamiento paso a paso
- **Role-based**: Prompt con una persona específica (experto en cine)
- **Structured**: Prompt que pide output en formato específico

**Uso**:
```python
from prompts import PROMPT_STRATEGIES

prompt_fn = PROMPT_STRATEGIES['zero_shot']
formatted_prompt = prompt_fn(conversation_context)
```

---

#### `evaluator.py`
**Propósito**: Calcula métricas de evaluación para recomendaciones.

**Métricas implementadas**:
- **Recall@K**: Proporción de items relevantes que fueron recomendados
- **Precision@K**: Proporción de items recomendados que son relevantes
- **NDCG@K**: Normalized Discounted Cumulative Gain (considera ranking)
- **Hit Rate@K**: Si al menos 1 item relevante está en top-K
- **MRR**: Mean Reciprocal Rank del primer item relevante

**Características**:
- Fuzzy matching para manejar variaciones en nombres de películas
- Normalización de títulos (remueve años, puntuación, etc.)
- Umbral de similitud configurable

**Uso**:
```python
from evaluator import RecommendationEvaluator

evaluator = RecommendationEvaluator(fuzzy_match_threshold=0.85)
metrics = evaluator.evaluate_all(recommended, ground_truth, k_values=[5, 10])
```

---

#### `run_experiment.py`
**Propósito**: Pipeline principal para ejecutar experimentos completos.

**Funcionalidades**:
- Orquesta el pipeline completo: carga datos → carga modelo → evalúa → guarda resultados
- Calcula métricas agregadas (media, std, mediana, min, max)
- Guarda resultados en JSON con metadata del experimento
- Maneja errores y memoria GPU
- Barra de progreso para seguimiento

**Uso**:
```python
from run_experiment import run_evaluation

run_evaluation(
    model_name="google/gemma-2-2b-it",
    dataset_path=".",
    output_dir="./results",
    prompt_strategy='zero_shot',
    n_samples=None,  # todas las muestras
    load_in_8bit=False,
    context_ratio=0.7,
    temperature=0.7
)
```

**Output**: Archivo JSON en `./results/` con:
- Información del experimento
- Métricas agregadas
- Ejemplos de resultados individuales

---

#### `experiments_midterm.py`
**Propósito**: Script para ejecutar batch de experimentos para el Midterm.

**Configuración**:
Define lista de experimentos a ejecutar en secuencia:
- Diferentes modelos (Gemma, Llama, Qwen)
- Diferentes tamaños (2B, 3B, 9B)
- Diferentes estrategias de prompting
- Configuraciones de cuantización

**Uso**:
```bash
python experiments_midterm.py
```

⚠️ **Advertencia**: Toma varias horas ejecutar todos los experimentos.

---

---

## 🚀 Guía de Uso Rápido

### 1. Instalación
```bash
# Instalar dependencias
pip install transformers torch
```
---

## 📊 Formato de Resultados

Los resultados se guardan en `./results/` como archivos JSON:
```json
{
  "experiment_info": {
    "model_name": "google/gemma-2-2b-it",
    "prompt_strategy": "zero_shot",
    "n_samples_evaluated": 523,
    "context_ratio": 0.7,
    "timestamp": "20251027_123456"
  },
  "aggregated_metrics": {
    "recall@10_mean": 0.0288,
    "recall@10_std": 0.1523,
    "precision@10_mean": 0.0038,
    "ndcg@10_mean": 0.0135,
    "hit_rate@10_mean": 0.0385,
    "mrr_mean": 0.0092,
    "latency_mean": 9.05
  },
  "sample_results": [...]
}
```

---

## 🔬 Metodología

### Pipeline de Evaluación
```
1. PREPROCESAMIENTO
   ↓
   - Cargar conversaciones ReDial
   - Dividir en contexto (70%) y ground truth (30%)
   - Filtrar conversaciones con ground truth válido

2. GENERACIÓN
   ↓
   - Cargar LLM desde HuggingFace
   - Formatear prompt según estrategia
   - Generar lista de 10 recomendaciones

3. EXTRACCIÓN
   ↓
   - Parsear respuesta del LLM
   - Extraer títulos de películas
   - Limpiar formato

4. EVALUACIÓN
   ↓
   - Fuzzy matching con ground truth
   - Calcular métricas (Recall, Precision, NDCG, etc.)
   - Medir latencia

5. AGREGACIÓN
   ↓
   - Promediar métricas sobre todas las muestras
   - Calcular estadísticas (mean, std, median)
   - Guardar resultados
```

### Context Ratio

El `context_ratio` define qué proporción de la conversación se usa como contexto:
- **0.7** = 70% de mensajes como contexto, 30% para ground truth
- **0.6** = 60% de mensajes como contexto, 40% para ground truth

Ratios más bajos → más ground truth disponible → más muestras válidas

---

## 📈 Métricas Clave

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| **Recall@10** | % de películas relevantes que fueron recomendadas | Qué tan completo es el set de recomendaciones |
| **Precision@10** | % de recomendaciones que son relevantes | Qué tan preciso es el set de recomendaciones |
| **NDCG@10** | Calidad del ranking (penaliza items relevantes mal rankeados) | Qué tan bien ordenadas están las recomendaciones |
| **Hit Rate@10** | % de conversaciones con al menos 1 acierto | Éxito básico del sistema |
| **MRR** | Reciprocal rank del primer acierto | Qué tan arriba aparece el primer acierto |
| **Latency** | Tiempo de generación (segundos) | Eficiencia computacional |

---