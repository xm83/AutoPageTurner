from .rectangle import Rectangle

note_step = 0.0625
# note_step = 0.073

note_defs = {
     -4 : ("g6", 79),
     -3 : ("f6", 77),
     -2 : ("e6", 76),
     -1 : ("d6", 74),
      0 : ("c6", 72),
      1 : ("b5", 71),
      2 : ("a5", 69),
      3 : ("g5", 67),
      4 : ("D5", 65),
      5 : ("e5", 64),
      6 : ("d5", 62),
      7 : ("c5", 60),
      8 : ("B4", 59),
      9 : ("a4", 57),

     10 : ("A4", 55),
     11 : ("f4", 53),
     12 : ("G4", 52),# 12 : ("e4", 52),
     13 : ("d4", 50),
     14 : ("c4", 48),
     15 : ("b3", 47),
     16 : ("a3", 45),
     17 : ("f3", 53),


     # -4 : ("g5", 79),
     # -3 : ("f5", 77),
     # -2 : ("e5", 76),
     # -1 : ("d5", 74),
     #  0 : ("c5", 72),
     #  1 : ("b4", 71),
     #  2 : ("a4", 69),
     #  3 : ("g4", 67),
     #  4 : ("f4", 65),
     #  5 : ("e4", 64),
     #  6 : ("d4", 62),
     #  7 : ("c4", 60),
     #  8 : ("b3", 59),
     #  9 : ("a3", 57),
     # 10 : ("g3", 55),
     # 11 : ("f3", 53),
     # 12 : ("e3", 52),
     # 13 : ("d3", 50),
     # 14 : ("c3", 48),
     # 15 : ("b2", 47),
     # 16 : ("a2", 45),
     # 17 : ("f2", 53),
    
}

class Note(object):
    def __init__(self, rec, sym, staff_rec, sharp_notes = [], flat_notes = []):
        self.rec = rec
        self.sym = sym

        middle = rec.y + (rec.h / 2.0)
        height = (middle - staff_rec.y) / staff_rec.h
        note_def = note_defs[int(height/note_step + 0.5)]
        self.note = note_def[0]
        self.pitch = note_def[1]
        if any(n for n in sharp_notes if n.note[0] == self.note[0]):
            self.note += "#"
            self.pitch += 1
        if any(n for n in flat_notes if n.note[0] == self.note[0]):
            self.note += "b"
            self.pitch -= 1


