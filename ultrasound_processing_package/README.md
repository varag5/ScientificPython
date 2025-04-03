# ScientificPython

Ultrasound Processing Package

Ez egy egyszerű Python-csomag, amit ultrahangos képek feldolgozására készült. Jelenleg két fő dologra alkalmas:

Zaj eltávolítása és kontúrok kiemelése ultrahangos képekről

Íves ultrahang képek átalakítása sík képpé (pl. amikor a kép ívelt alakban készült, de síkban szeretnénk látni)

Telepítés

Ha helyben szeretnéd használni, egyszerűen klónozd le, majd futtasd ezt:

pip install ultrasound_processing_pacage

Hogyan használd

1. Zajeltávolítás és kontúrkiemelés

Itt egy példa, hogy kell zajt eltávolítani és kontúrt kiemelni egy képről:

import ultrasound_processing_package as upp
import matplotlib.pyplot as plt

# Saját képed útvonala
kep_ut = "data/transformation_curved_flat_whole_test.png"

eredeti, kuszob, maszk, maszkolt_kep = upp.remove_top_noise_and_keep_first_white(kep_ut)

plt.imshow(maszkolt_kep, cmap='gray')
plt.title("Zajeltávolítás után")
plt.axis('off')
plt.show()

2. Íves ultrahang-kép átalakítása

Ha íves ultrahang képet akarsz síkra transzformálni:

import ultrasound_processing_package as upp
import matplotlib.pyplot as plt

kep_ut = "data/sample_curved_cropped_01.png"

sik_kep = upp.transform_curved_to_flat(kep_ut)

plt.imshow(sik_kep, cmap='gray')
plt.title("Íves ultrahang síkban")
plt.axis('off')
plt.show()

Mi kell hozzá?

Python 3.6 vagy újabb, valamint:

NumPy

SciPy

OpenCV (opencv-python)

Matplotlib

Pillow

Ezeket a csomagokat amúgy automatikusan telepíti legjobb esetbe, amikor felrakod a csomagot.
