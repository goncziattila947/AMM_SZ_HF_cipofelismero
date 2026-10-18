# Cipőfelismerő Mélytanulási Projekt

## Alapfogalmak és Szakszavak

* **Mélytanulási osztályozó modell (Deep Learning Classifier):** Olyan mesterséges neurális hálózat, amely rétegeken keresztül megtanulja felismerni a képek vizuális mintázatait (vonalak, logók, formák), majd a képet besorolja egy adott kategóriába (pl. Adidas, Nike vagy Converse).
* **Tanuló adathalmaz (Train set):** A modell „tankönyve”. Olyan képek halmaza, ahol a modell tudja a helyes választ, és ezekből próbálja megtanulni az összefüggéseket.
* **Címke / Osztálycímke (Label):** A helyes válasz az adott képre (pl. Nike vagy Férfi).
* **Teszt adathalmaz (Test set):** A „vizsgadolgozat”[cite: 5]. Új, a tanítás során nem látott képek gyűjteménye, ahol nincs megadva a megoldás; a betanított modellnek önállóan kell megmondania a címkét.
* **Keresztvalidáció (Cross-validation):** A tanuló adatokat több részre (pl. 5 egyenlő szeletre) osztjuk[cite: 5]. Felváltva 4 részből tanul a modell, és az 1 kimaradó részen ellenőrizzük a pontosságát. Ezzel kiszűrhető, hogy a modell csak a véletlen szerencse miatt teljesített-e jól.
* **Paraméteroptimalizálás / Hiperparaméter-hangolás:** A modell működését szabályozó „tekerőgombok” (pl. tanulási sebesség, kötegnagyság, rétegek száma) beállítása úgy, hogy a lehető legpontosabb legyen a végeredmény.
* **Jósági mutatók (Metrics):** Százalékos vagy statisztikai mérőszámok, mint az Accuracy (összes esetből hányat talált el jól) és az F1-score (kiegyensúlyozott pontossági mutató).
* **Annotáció:** Kézi felcímkézés[cite: 5]. A férfi/női feladathoz a képeket meg kell nyitni, és kézzel rá kell írni az adathalmazra, hogy az adott cipő férfi vagy női modell.
* **Transzfer tanulás (Transfer Learning):** Óriási adatbázisokon (pl. ImageNet) előre betanított neves hálózatok (ResNet, EfficientNet) újrahasznosítása. 600 kép mélytanuláshoz nagyon kevés, így ennek használata gyakorlatilag elengedhetetlen a jó eredményhez.

---

## Technológiai Stack és Eszközök

| Terület | Ajánlott eszköz |
| :--- | :--- |
| **Fejlesztői környezet** | Google Colab |
| **Keretrendszer** | PyTorch vagy TensorFlow / Keras |
| **Kézi annotálás** | Mappaelosztás szerint |
| **Adatbővítés** | Albumentations vagy torchvision.transforms |

---

## A feladat felépítése lépésről lépésre

### 1. Adatelőkészítés és annotáció
* **1. feladat (Márka):** A 600 kép már mappákba van rendezve (Adidas/, Converse/, Nike/)[cite: 5]. Ezeket be kell olvasni és méretre vágni (pl. 224x224 képpont).
* **2. feladat (Férfi/Női):** Osszátok szét a 600 képet a csapattagok között, és egy táblázatban jelöljétek be mindegyikre, hogy férfi vagy női (a feladat kiemeli: unisex nem lehet, dönteni kell). Dokumentáljátok a döntési szabályaitokat (pl. színek, fazon, sarokmagasság alapján döntöttetek-e).

### 2. Képbővítés (Data Augmentation)
A 600 kép kevés, ezért tanítás közben alkalmazzatok horizontális tükrözést, kis mértékű forgatást és színkorrekciót, hogy a háló ne magolja be a képeket (ne lépjen fel túlilleszkedés / overfitting).

### 3. Modellválasztás és tanítás (Transfer Learning)
* Válasszatok egy bevált előtanított modellt (pl. EfficientNet-B0 vagy ResNet50).
* Cseréljétek le az utolsó osztályozó réteget:
  * A márkafelismerésnél egy **3 kimenetű** rétegre (Adidas, Converse, Nike).
  * A nem felismerésnél egy **2 kimenetű** rétegre (Férfi, Női).
* Tanítsátok be mindkét feladatot 5-szörös keresztvalidációval (5-Fold Cross Validation).

### 4. Kísérletezés és kiértékelés
* Próbáljatok ki legalább 2 különböző architektúrát (pl. ResNet vs. MobileNet vagy EfficientNet), és hasonlítsátok össze az eredményeket táblázatban.
* Készítsetek tévesztési mátrixot (Confusion Matrix): ez vizuálisan megmutatja, hogy melyik márkát mivel keverte össze a rendszer.

### 5. Teszt előrejelzés és dokumentáció
* Futtassátok le a két végleges modellt a tanító címkék nélküli teszt halmazon, és mentsétek el a kapott osztálycímkéket az előírt formátumban.
* Írjátok meg a beszámolót: mutassátok be a csapatmunkát, az annotáció szempontjait, a kipróbált modellek összehasonlítását és a levont tanulságokat.

---

## Első rész: Adatelőkészítés és Annotációs Szabályok

* **Képméret:** Az összes kép 240x240 méretű.
* **Módszer:** Manuálisan címkézni kell különböző megbeszélt szabályok alapján.
* **Arányok:** Törekedni kell a nagyjából 50-50%-os arányra, mert a kiegyensúlyozatlan adathalmazon a neurális háló elfogulttá válhat.

### Női kategória
* Karcsúsított, szűkített orr-rész, kecsesebb sziluett.
* Emelt talp, platform (vastagított talp), magassarkú jellegű vagy balerina-jellegű sneakerek.
* Mélyebb kivágás a boka körül, vékonyabb nyelv és fűző[cite: 5].
* Pasztell árnyalatok (rózsaszín, babakék, menta, lila), csillogó/metál elemek, virágminták, arany/rózsaarany részletek.

### Férfi kategória
* Széles, masszív talpszerkezet (különösen a sarokrésznél).
* Robusztus, szögletesebb orrforma és szélesebb felsőrész.
* Masszív futó- vagy kosárlabdacipő felépítés, vastag párnázottság.
* Sötét, tompa színek (katonai zöld, földszínek, mélykék, antracit), neonszínekkel kombinált technikai fekete.

### Unisex eset kezelése
* **Arányok és méretérzet:** A képen látható cipő szélesnek, nagynak tűnik (42-45-ös méret arányai), vagy inkább keskeny, apróbb (36-39-es méret)?
* **Szabály:** Nem lehet unisex, muszáj vagy női vagy férfi kategóriát választani!
