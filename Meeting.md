# 🧠 Notas para la reunión con el profesor – Clustering de Consumo Eléctrico

## ✅ Decisiones tomadas

### 🔹 Clustering sobre consumo eléctrico de 24h (por día)

He decidido hacer el clustering a partir del **perfil diario de consumo eléctrico** (es decir, 24 valores por día).  
Esto me permite trabajar con datos homogéneos y comparables entre viviendas.

> ❌ **Descarté** hacer clustering sobre toda la serie temporal de cada vivienda (CUPS), ya que:
- Las longitudes de las series varían mucho entre viviendas.
- Los datos no son homogéneos (algunos hogares tienen más días, otros menos).
- Probé con DTW (Dynamic Time Warping) pero **no da buenos resultados** en este contexto.

### 🔹 Clustering por estación o por diás concretos

Por ejemplo: Se puede refinar el análisis haciendo clustering **por temporadas**:
- Sólo días de invierno.
- Sólo días de verano.
- Esto puede revelar patrones específicos del uso energético estacional (ej. calefacción vs. aire acondicionado).



### 🔎 ¿Qué es ISAC? y Porque escogí K-ISAC-TLP

Este análisis se basa en la técnica de las curvas de **ISAC** (*Inflection Stability Area Criterion*), una metodología diseñada para seleccionar el mejor número de clusters (**k**) a partir de la geometría de curvas de evaluación, como el error o la cantidad de clusters irrelevantes.

**ISAC** es un enfoque **geométrico-visual** para detectar el valor óptimo de k utilizando curvas como:

- **Error vs. k** (por ejemplo, MAE, SSE, etc.)
- **Cantidad de clusters vacíos o irrelevantes vs. k**

Se basa en dos ideas clave:

- **Área de triángulos**: mide cuánto cambia la forma de la curva. Un área pequeña sugiere que la curva ya se ha estabilizado.
- **Pendiente**: indica la ganancia marginal. Si la pendiente se vuelve plana (poca mejora), probablemente ya no vale la pena aumentar k.

> 🧠 *El principio de ISAC es:*  
> *"El mejor k es el punto donde la curva se aplana y deja de mejorar significativamente."*

Esto recuerda al **método del codo**, pero ISAC lo automatiza utilizando **triángulos móviles** sobre la curva.

---

## 📊 Ejemplo práctico con datos de consumo energético

Supongamos que estamos aplicando k-means a perfiles diarios de consumo de viviendas. Para varios valores de k (de 2 a 10), calculamos:

- `mae_values`: el error medio absoluto entre perfiles reales y los centroides.
- `irrelevant_clusters`: número de clusters con muy poca población o comportamiento indistinto (por ejemplo, menor al 3% del total o patrones planos).

```python
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
mae_values = [0.35, 0.28, 0.23, 0.21, 0.20, 0.195, 0.193, 0.192, 0.191]
irrelevant_clusters = [0, 0, 0, 1, 1, 2, 2, 3, 4]
```
---

## 🤔 Dudas y reflexiones sobre el escalado

### 🔸 Clustering con `Aggregated Features` solamente

He probado usar features agregadas (ej. media, varianza, skewness, total, etc.).  
Para que funcionen bien en KMeans, es necesario aplicar **escalado**, por ejemplo con `StandardScaler`.

Pero tengo dudas:
- Escalar transforma los valores del consumo real.
- ¿Tiene sentido hacer esto si pierdo la magnitud real del consumo?
- Además, si quisiera usar modelos como **K-ISAC-TLP** o métricas como MAE, el escalado rompe el significado físico del dato.

📌 **Pregunta al profesor:**  
> ¿Debería descartar completamente el uso de aggregated features si requieren escalado?  
> ¿O hay una forma de mantener su interpretación sin distorsionar el consumo real?

---

### 🔸 Clustering combinado: aggregated features + perfil diario (24h)

Pensé en combinar ambos tipos de datos (agregados + 24h), pero:
- Requiere escalar ambas partes para que estén en la misma magnitud.
- Esto **distorsiona el valor real del consumo**, y puede hacer que la parte de 24h pierda peso en la agrupación.

Por esta razón, **descarté esta opción también**.

Se propuso una solución alternativa:  
> Hacer un **ensemble de modelos de clustering**, uno para aggregated features y otro para consumo diario.

Pero:
- Un ensemble **no es un modelo único**, sino una combinación.
- No sé si este enfoque es coherente con los objetivos del trabajo.

📌 **Pregunta al profesor:**  
> ¿Es válido plantear un ensemble de clusters (agregados + diarios)?  
> ¿O deberíamos buscar una forma de integrar todo en un solo modelo?

---

## ❌ Otras ideas descartadas

- **Aggregated features a nivel global (ej. sumar todo el año o dos años):**  
  Esto pierde completamente la granularidad diaria. No tiene sentido hacer un promedio de dos años cuando lo interesante son los patrones diarios.

---

## 📌 Conclusión

### ✅ Me quedo con:
- Clustering de consumo **eléctrico diario (24h)**.
- Opción de dividir por estaciones (invierno, verano).

### ❓ Lo que quiero saber:
- ¿Es suficiente con el enfoque actual?
- ¿Tiene sentido explorar más ideas (ensemble, otras variables)?
- ¿Me conviene dejar de lado del todo las aggregated features?

