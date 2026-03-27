# PyTorch Layer-Übersicht

## Convolutional Layer (Feature-Extraktion)

| Layer | Was es macht |
|-------|-------------|
| **`nn.Conv2d`** | Filter gleiten über das Bild, erkennen Muster (Kanten, Texturen, etc.). |
| **`nn.Conv1d`** | Dasselbe, aber für 1D-Daten (z.B. Audiosignale, Zeitreihen) |
| **`nn.Conv3d`** | Dasselbe, aber für 3D-Daten (z.B. Videos, medizinische 3D-Scans) |
| **`nn.ConvTranspose2d`** | "Umgekehrte" Convolution — vergrössert das Bild statt es zu verkleinern. Wird bei Bildgenerierung verwendet (z.B. GANs) |

## Pooling Layer (Verkleinerung)

| Layer | Was es macht |
|-------|-------------|
| **`nn.MaxPool2d`** | Nimmt den **grössten** Wert aus jedem Fenster. |
| **`nn.AvgPool2d`** | Nimmt den **Durchschnitt** aus jedem Fenster. Glättet stärker als MaxPool. |
| **`nn.AdaptiveAvgPool2d`** | Wie AvgPool, aber du gibst die **Zielgrösse** an statt die Fenstergrösse. `AdaptiveAvgPool2d(1)` → reduziert jede Feature Map auf 1×1 (ein einzelner Wert). Wird in modernen Architekturen (ResNet etc.) statt Flatten verwendet. |

## Aktivierungsfunktionen (Nicht-Linearität)

| Layer | Was es macht |
|-------|-------------|
| **`nn.ReLU`** | Negative Werte → 0, positive bleiben. Standard-Wahl. |
| **`nn.LeakyReLU`** | Wie ReLU, aber negative Werte werden nicht auf 0 gesetzt sondern nur stark verkleinert (z.B. × 0.01). Verhindert "tote Neuronen". |
| **`nn.GELU`** | Glattere Version von ReLU. Wird in Transformern verwendet (GPT, BERT). |
| **`nn.Sigmoid`** | Quetscht Werte in den Bereich [0, 1]. Für binäre Klassifikation (ja/nein). |
| **`nn.Tanh`** | Quetscht Werte in den Bereich [-1, 1]. |
| **`nn.Softmax`** | Wandelt Werte in Wahrscheinlichkeiten um (summieren sich zu 1.0). Nicht explizit nötig — `CrossEntropyLoss` macht das intern. |

## Fully Connected (Klassifikation)

| Layer | Was es macht |
|-------|-------------|
| **`nn.Linear`** | Jeder Input-Wert wird mit jedem Output-Wert verbunden. `Linear(4096, 128)` = 4096 Inputs × 128 Outputs. |
| **`nn.Flatten`** | Formt den mehrdimensionalen Tensor in 1D um. `[64, 8, 8]` → `[4096]`. Nötig vor dem ersten Linear-Layer. |

## Normalisierung (Training stabilisieren)

| Layer | Was es macht |
|-------|-------------|
| **`nn.BatchNorm2d`** | Normalisiert die Werte **pro Batch** auf Mittelwert≈0 und Standardabweichung≈1. Beschleunigt das Training massiv und stabilisiert es. |
| **`nn.LayerNorm`** | Normalisiert **pro Sample** statt pro Batch. Standard in Transformern. |
| **`nn.GroupNorm`** | Kompromiss zwischen BatchNorm und LayerNorm. Funktioniert auch mit kleinen Batch Sizes. |

## Regularisierung (Overfitting verhindern)

| Layer | Was es macht |
|-------|-------------|
| **`nn.Dropout`** | Setzt zufällig einen Anteil der Werte auf 0 während des Trainings. `Dropout(0.5)` = 50% werden deaktiviert. Zwingt das Netzwerk, redundante Features zu lernen. |
| **`nn.Dropout2d`** | Wie Dropout, aber deaktiviert **ganze Feature Maps** statt einzelner Werte. Besser geeignet nach Conv-Layern. |

## Recurrent (Sequenzen)

| Layer | Was es macht |
|-------|-------------|
| **`nn.RNN`** | Verarbeitet Sequenzen (Text, Zeitreihen). Hat ein "Gedächtnis" für vorherige Schritte. |
| **`nn.LSTM`** | Wie RNN, aber mit besserem Langzeitgedächtnis. Standard für Sequenz-Aufgaben. |
| **`nn.GRU`** | Wie LSTM, aber einfacher und schneller. Oft gleichwertig. |

## Embedding (für diskrete Daten)

| Layer | Was es macht |
|-------|-------------|
| **`nn.Embedding`** | Wandelt Ganzzahlen (z.B. Wort-IDs) in lernbare Vektoren um. Grundbaustein von NLP-Modellen. |

## Container (Layer gruppieren)

| Layer | Was es macht |
|-------|-------------|
| **`nn.Sequential`** | Führt Layer nacheinander aus: `nn.Sequential(Conv2d, ReLU, MaxPool, ...)` |

## Parameter-Berechnung

**Conv2d:** `(kernel × kernel × in_channels + 1) × out_channels` — das +1 ist der Bias pro Filter.

**Linear:** `(in_features + 1) × out_features` — das +1 ist der Bias pro Neuron.

**ReLU, MaxPool, Flatten, Dropout, BatchNorm (fast):** haben keine/kaum lernbare Parameter.
