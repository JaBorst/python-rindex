Implementierung eines Moduls zur \textbf{W}ord \textbf{E}mbedding \textbf{Vis}ualisuerung. Man kann ein Word Embedding in Form einer Python-\texttt{dict} Datenstruktur laden und danach Methoden zur Dimensionsreduktion und Visualisierung anwenden. Hier eine kurze Übersicht über die wichtigsten Funktionen:

'''
load_model(filename = "", norm = False):
'''
Lädt ein  Word Embedding aus einer Datei. Das Word Embedding soll hier ein mit \texttt{pickle} gedumptes Dictionary sein. Wenn \texttt{norm} gesetzt wird wird das Wordembedding feature-weise genormt.

\begin{lstlisting}[language=Python]
	dim_reduce( self, method = "tsne", 
			  target_dim = 2, 
			  points = None, 
			  metric = "minkoswki"):
\end{lstlisting}
	Mit dieser Funktion lässt sich die Dimension des Embeddings reduzieren. Dazu kann mit \texttt{points} entweder eine Menge an Punkten gegeben werden, ohne Punkte wird versucht das komplette Embedding zu reduziern ( sehr rechenaufwändig).\\
	Die zur Verfügung stehenden Methoden sind TSNE, Truncated SVD, Spectral(Laplacian Eigenmaps), Isomap, Locally Linear Embedding (lle), KPCA. Die Implementierung sind aus \texttt{scipy} entnommen.
	Isomap wurde modifiziert und kann mit verschiedenen Metriken ausgeführt werden.
\begin{lstlisting}[language=python]
def wordpairs(self, words = testpairs, 
			testaddwords=addWords, 
			method = "tsne", 
			metric = "euclidean", 
			save= False, show = True):
\end{lstlisting}
Mit dieser Funktion kann man eine Liste von Wortpaare spezifizieren, die erst mit der gegeben \texttt{method} reduziert werden und dann geplottet werden. Zusätzlich kann man ein \texttt{dict} übergeben, dass eine Liste von \glqq positive\grqq und eine Liste \glqq negative\grqq Wörter enthält:

\begin{verbatim}
testaddwords = {'positive' = ["Frau"], 'negative' = ["Mann"]\right\rbrace }
\end{verbatim}

Wenn \texttt{testaddwords} spezifiziert wurde dann wird von jedem ersten Eintrag der Wortpaare die negativen Einträge abgezogen und die Positiven addiert. Das Ergebnis wird mit dem zweiten Element der Wortpaare verglichen und der Ähnlichkeits-Rang mit in den Plot eingetragen
\begin{lstlisting}[language=python]
def wordlist(self, words = [], 
		method = "tsne", metric = "manhattan"):
\end{lstlisting}
Mit dieser Funktion können einfache Wortlisten reduziert und ein 2D-Plot erstellt werden. Die Funktion sammelt dazu zuerst die Vektoren und reduziert nur diese Teilmenge.
\begin{lstlisting}[language=python]
def wordlistwhole(self, words = [],
		method = "tsne", metric = "manhattan"):
\end{lstlisting}
Diese Funktion verhält sich ähnlich wir \texttt{wordlist}, allerdings wird versucht zu erst das gesamte Embedding zu reduzieren und dann die Vektoren der \texttt{words} zu sammeln und zu plotten. Konnte auf Grund des etwas höheren Rechenaufwands nicht ausgiebig getestet werden.

\subsection{Beispiele}
Es wurde auf Basis von 3 Millionen deutscher Sätze aus dem Deutschen Wortschatz ein Modell erstellt mit \texttt{word2vec}, \texttt{glove} und \texttt{rindex}. 
Die Zahl im Plot gibt für jedes Wortpaar einen Rang an. Berechnet wurde dieser in dem zum Beispiel $v('K\textnormal{ö}nig') +v('Frau') - v('Mann')$ gerechnet wurde und dann die Ähnlichkeit des sich ergebenden Vektor zu allen anderen Worten berechnet wurde und ausgegeben wurde auf welchem Rang $K\textnormal{ö}nigin$ stand.
Die Funktion \texttt{wordpairs} wurde mit verschiedenen Dimensionsreduktionsmethoden getestet und die Ergebnisse verglichen.

\begin{figure}[ht]
	\centering
	\includegraphics[width=\iwidth\linewidth]{plots/rindex/RIndex-tsne}
	\includegraphics[width=\iwidth\linewidth]{plots/w2v/W2V-tsne}
	\includegraphics[width=\iwidth\linewidth]{plots/glove/Glove-tsne}
	\caption{Vergleich dreier Word Embeddings, die mit TSNE reduziert wurden}
	\label{fig:rindex-tsne}
\end{figure}
TSNE ist ein auf Dimensionsreduktionsverfahren, das auf gewichteten euklidischen Abständen basiert. 
Die Plots in allen drei Fällen keine klar erkennbare Struktur.

\begin{figure}[ht]
	\centering
	\includegraphics[width=\iwidth\linewidth]{plots/rindex/RIndex-truncated_svd}
	\includegraphics[width=\iwidth\linewidth]{plots/w2v/W2V-truncated_svd}
	\includegraphics[width=\iwidth\linewidth]{plots/glove/Glove-truncated_svd}
	\caption{Vergleich dreier Word Embeddings, die mit truncated\_svd reduziert wurden}
	\label{fig:rindex-svd}
\end{figure}

Im Weiteren wurden graph-basierte Dimensionsreduktions-Verfahren, wie Locally Linear Embedding und Isomap, getestet.

\begin{figure}[ht]
	\centering
	\includegraphics[width=\iwidth\linewidth]{plots/rindex/RIndex-lle}
	\includegraphics[width=\iwidth\linewidth]{plots/w2v/W2V-lle}
	\includegraphics[width=\iwidth\linewidth]{plots/glove/Glove-lle}
	\caption{Vergleich dreier Word Embeddings, die mit Locally Linear Embedding reduziert wurden}
	\label{fig:rindex-lle}
\end{figure}

Isomap wurde ebenfalls für verschiedene Metriken getestet. 
\begin{figure}[ht]
	\centering
	\includegraphics[width=\iwidth\linewidth]{plots/rindex/RIndex-isomap(manhattan)}
	\includegraphics[width=\iwidth\linewidth]{plots/w2v/W2V-isomap(manhattan)}
	\includegraphics[width=\iwidth\linewidth]{plots/glove/Glove-isomap(manhattan)}
	\caption{Vergleich dreier Word Embeddings, die mit Isomap und Manhattan-Metrik reduziert wurden}
	\label{fig:rindex-iso}
\end{figure}
Bei \texttt{Word2Vec} und Isomap mit Manhattan-Metrik in Abbildung \ref{fig:rindex-iso} lässt sich hier relativ gut eine Struktur erkennen, die nahe legt, dass die Vektoraddition funktionieren kann.

\begin{figure}[ht]
	\centering
	\includegraphics[width=\iwidth\linewidth]{plots/rindex/RIndex-isomap(minkowski)}
	\includegraphics[width=\iwidth\linewidth]{plots/w2v/W2V-isomap(minkowski)}
	\includegraphics[width=\iwidth\linewidth]{plots/glove/Glove-isomap(minkowski)}
	\caption{Vergleich dreier Word Embeddings, die mit Isomap und Minkowski-Metrik reduziert wurden}
	\label{fig:rindex-iso}
\end{figure}
