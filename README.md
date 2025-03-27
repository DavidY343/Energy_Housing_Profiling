# TFG_electric_clustering

El id "c1244d6dea7a" tiene la hora con dos ceros de mas:
c1244d6dea7a;02/08/2023;24:00;0,001;;



Cosas que he ido descubriendo de los datos:

Las horas estaban a veces mal formateadas por ejemplo "1:00" en vez de "01:00" o "24:00:00" en vez de "24:00"
Existian dos formatos de fechas que he tenido que agrupar
Los datos de la serie temporal dependen de la vivienda algunso empiezan el 07-08-2021 y otros el 30-08-2021. Los he puesto todos a emepzar a partir del 30 para unificar
04666163609d Consumo 07-08-2021_07-08-2023 Este cups no acaba el 7, acaba el 5
03c8338d7f1d Consumo 07-08-2021_07-08-2023 Este cups no acaba el 7, acaba el 4
277f70c6024c Consumo 07-08-2021_07-08-2023 Este cups no empiza en el 7-08 empiza en el 12-05

Cosas que comentar a agapito:

A dia de hoy mi idea del clustering va ser pasar todo a formato ancho y hacer clustering de eso. 
Otra idea que he visto es solo hacer clustering por ejemplo de dias. Cojo los domingos y hago clustering de los domingos de todas las viviendas.