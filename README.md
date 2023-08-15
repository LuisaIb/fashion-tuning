# shoe-tuning
Hyperparameter Tuning of different Machine Learning models for image classification of shoe dataset.

---
Dataset: https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged?resource=download

---
**Aufgabenstellung:**<br>
<ul>
  <li>Durch Gruppe gewählte Problemstellung aus Bereich Machine Learning (Datensatz, Modell, Parameter)</li>
  <li>Problemstellung lösen, mit empirischer Untersuchung, Einsatz der Ansätze aus der Vorlesung (Parametertuning!)</li>
  <li>Präsentation: Beschreibung von</li>
  <ul>
    <li>Problemstellung / Ausgangslage</li>
    <ul>
      <li>Auch Modell, aber Schwerpunkt auf die zu tunenden Parameter</li>
      <li>Was ändern diese? Erwarteter Einfluss?</li>
    </ul>
    <li>Lösung / Methoden</li>
    <li>Empirische Ergebnisse</li>
    <ul>
      <li>Schwerpunkt: welchen Einfluss hatten welche Parameter tatsächlich?</li>
    </ul>
    <li>Bewertung und Schlussfolgerungen</li>
  </ul>
</ul>

---

**Besprechung - 15.08.2023:**<br>
Weitere Planung / Agenda - Präsentation:
<ul>
  <li>Problemstellung / Ausgangslage: Was ist FashionMNIST? Woher kommen die Daten? -> Zalando -> Benchmark Datensatz -> s. Paper zu Datensatz (+ Quellen) => Image Classiication + multi class (non binary) -> vllt. auf Metriken eingehen</li>
  <li>EDA (5min), Größe Datensatz, beispielhafte Bilder, wie Aufbau von Datensatz, Balance der Klassen, Pixel Angaben nennen</li>
  <li> Data Preparation erwähnen</li>
  <li>Metriken (Accuracy (vllt. Balanced Acc), Loss Function (CrossEntropy)) nennen (evtl. + Laufzeit evtl. von Tuning / Inferenz als Begründung)</li>
  <li>CNN Model vorstellen, warum CNN genommen, vllt. Architektur </li> 
  <li>Welche Parameter gibt es? Welche Auswirkungen haben diese? Was könnte bei Anpassungen dieser passieren?
  <ul>
    <li>Neural Architecture Search: optimaler Aufbau finden</li>
    <li>klassischere Hyperparameter: Konfiguration der Parameter, die entscheidend für Model-Training sind, bestimmen</li>
  </ul>
  </li>
  <li>konzentrieren Hyperparameter Tuning, weil Architecture Search ressourcen aufwändig, in mehreren Paper für NAS gute Ergebnisse bereits gefunden (generell: nicht so klar voneinander trennbar) </li>
  <li>betrachtete Parameter vorstellen: lr, epochs, batchsize - vllt. weitere Parameter auch nennen wie dropout, optimizer</li>
  <li>Epochs, LR, batchsize genauer vorstellen & Einflüsse nennen zusammen mit gewählten Parameter Ranges</li>
  <li>initiale Architektur vorstellen mit Grafik (vllt. aus Paper mit gutem NAS) -> d.h. Begründung, mit default Werten (Industrie Standard) initiale Architektur durchlaufen gelassen - Ergebnis vorstellen -> Kann man das besser oder schlechter machen?</li>
  <li>Hardware nennen, Rechenzeit festlegen und technische Vorgehensweise (Tensorboard, Tune, RandomSeed etc.)</li>
  <li>die durchgelaufenen Trials vorstellen, Grund warum weiter gehen -> RandomSearch in Anbetracht der Laufzeit / Hardware Ressourcen sehr ineffizient</li>
  <li>Hardware Ressourcen besser zu verwenden? -> Scheduler</li>
  <li>Zwischenergebnis von Trials in Laufzeit mit Scheduler und ohne vorstellen </li>
  <li>alternative Möglichkeit wäre jetzt: SMBO with Spot (cf. Zaeff, 2023)</li>
  <li>Grafik von Folien -> initiale Funktion von SMBO vorstellen -> Spot Package von Python</li>
  <li>gleiche Parameter + Surrogate Control (weitere Parameter einstellen) -> an default Werten orientiert</li>
  <li>auf Durchläufe eingehen, d.h. vllt. Anzahl anhand der Accuracy messen</li>
  <li>Einflüsse der Parameter anhand der Plots vorstellen</li>
  <li>mit Parallel Plot anfangen und dann ein Parameter rausgreifen (wenn auffällig)</li>
  <li>auf Konturplots abspringen - auf die erwarteten Ergebnisse zurückgehen -> 3 interessante Parameter rausgreifen</li>
  <li>Parameter Plots (kontur etc.) für die drei Tuning Durchläufe (bspw. auf eine Folie)
  <li>Boxplots von jedem Tuning beste Trial rausnehmen und plotten für die 2 Tunings + Default</li>
  <li>Bewertung und Schlussfolgerung: generelle / allgemeine Schlussfolgerungen auf NNs bezogen, Rechenleistung, Parameter (Architektur), etwas positives Richtung SMBO, etwas Richtung FashionMNIST, etw. über den Einfluss von den Parametern aus den Plots, Trade-Off zwischen Ergebnis und Rechenleistung, vllt. Takeaway Use Case bezogen (Zalando)</li> 
</ul>
