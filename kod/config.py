# config.py
# ... (pozostała część pliku zostaje bez zmian)

# Jeżeli Twoje AI Scene Score lubi się satururować na ~100%, to potrafi generować false-positive
# na materiałach REAL. Ten hook pozwala wyłączyć (wyzerować wagę) sceny w fuzji wyniku.
# GUI wywołuje to w _fuse_ai_score() jeżeli funkcja istnieje.

def should_suppress_scene(ai_face: float, ai_video: float) -> bool:
    """Suppress scene contribution when face/video signals exist.

    Heurystyka na mniej false-positive: jeśli model twarzy albo wideo dostarcza sygnał,
    ignorujemy scenę (bo często saturuje się do 100% na wielu filmach).
    """
    try:
        if ai_face is not None:
            return True
        if ai_video is not None:
            return True
    except Exception:
        pass
    return False
