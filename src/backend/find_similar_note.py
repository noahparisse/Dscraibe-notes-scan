from db import find_similar_note

print(find_similar_note("25/09/2025 Note 1 Changement relais 40kV > 20kV (secteur T4) Rappel: prévenir M. Martin (amateur: 0766376647) nouveau:", threshold=0.7))