# Wevis

## Funktionen

Implementierung eines Moduls zur *W*ord *E*mbedding *Vis*ualisuerung. Man kann ein Word Embedding in Form einer Python-_dict_ Datenstruktur laden und danach Methoden zur Dimensionsreduktion und Visualisierung anwenden. Hier eine kurze Übersicht über die wichtigsten Funktionen:

``` load_model(filename = "", norm = False): ```
Lädt ein  Word Embedding aus einer Datei. Das Word Embedding soll hier ein mit `pickle` gedumptes Dictionary sein. Wenn `norm` gesetzt wird wird das Wordembedding feature-weise genormt.

```dim_reduce( self, method = "tsne", 
			  target_dim = 2, 
			  points = None, 
			  metric = "minkoswki"):
```
Mit dieser Funktion lässt sich die Dimension des Embeddings reduzieren. Dazu kann mit \texttt{points} entweder eine Menge an Punkten gegeben werden, ohne Punkte wird versucht das komplette Embedding zu reduziern ( sehr rechenaufwändig).\\

Die zur Verfügung stehenden Methoden sind TSNE, Truncated SVD, Spectral(Laplacian Eigenmaps), Isomap, Locally Linear Embedding (lle), KPCA. Die Implementierung sind aus `scipy` entnommen.
Isomap wurde modifiziert und kann mit verschiedenen Metriken ausgeführt werden.

```
def wordpairs(self, words = testpairs, 
			testaddwords=addWords, 
			method = "tsne", 
			metric = "euclidean", 
			save= False, show = True):
```
Mit dieser Funktion kann man eine Liste von Wortpaare spezifizieren, die erst mit der gegeben \texttt{method} reduziert werden und dann geplottet werden. Zusätzlich kann man ein \texttt{dict} übergeben, dass eine Liste von "positive" und eine Liste "negative" Wörter enthält:

``` testaddwords = {'positive' = ["Frau"], 'negative' = ["Mann"] }
```
Wenn `testaddwords` spezifiziert wurde dann wird von jedem ersten Eintrag der Wortpaare die negativen Einträge abgezogen und die Positiven addiert. Das Ergebnis wird mit dem zweiten Element der Wortpaare verglichen und der Ähnlichkeits-Rang mit in den Plot eingetragen
```
def wordlist(self, words = [], 
		method = "tsne", metric = "manhattan"):
```

Mit dieser Funktion können einfache Wortlisten reduziert und ein 2D-Plot erstellt werden. Die Funktion sammelt dazu zuerst die Vektoren und reduziert nur diese Teilmenge.
```
def wordlistwhole(self, words = [],
		method = "tsne", metric = "manhattan"):
```
Diese Funktion verhält sich ähnlich wir \texttt{wordlist}, allerdings wird versucht zu erst das gesamte Embedding zu reduzieren und dann die Vektoren der \texttt{words} zu sammeln und zu plotten. Konnte auf Grund des etwas höheren Rechenaufwands nicht ausgiebig getestet werden.
