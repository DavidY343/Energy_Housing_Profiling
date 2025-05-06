# 🧠 Notas para la reunión con el profesor – Clustering de Consumo Eléctrico

## ✅ Decisiones tomadas

### 🔹 Clustering sobre consumo eléctrico de 24h (por día)

He decidido hacer el clustering a partir del **perfil diario de consumo eléctrico** (es decir, 24 valores por día).  
Esto me permite trabajar con datos homogéneos y comparables entre viviendas.

> ❌ **Descarté** hacer clustering sobre toda la serie temporal de cada vivienda (CUPS), ya que:
- Las longitudes de las series varían mucho entre viviendas.
- Los datos no son homogéneos (algunos hogares tienen más días, otros menos).
- Probé con DTW (Dynamic Time Warping) pero **no da buenos resultados** en este contexto.

### 🔹 Clustering por estación

Se puede refinar el análisis haciendo clustering **por temporadas**:
- Sólo días de invierno.
- Sólo días de verano.
- Esto puede revelar patrones específicos del uso energético estacional (ej. calefacción vs. aire acondicionado).

---

## 🤔 Dudas y reflexiones sobre el escalado

### 🔸 Clustering con `Aggregated Features` solamente

He probado usar features agregadas (ej. media, varianza, skewness, total, etc.).  
Para que funcionen bien en KMeans, es necesario aplicar **escalado**, por ejemplo con `StandardScaler`.

Pero tengo dudas:
- Escalar transforma los valores del consumo real.
- ¿Tiene sentido hacer esto si pierdo la magnitud real del consumo?
- Además, si quisiera usar modelos como **K-I-SAC-TLP** o métricas como MAE, el escalado rompe el significado físico del dato.

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
