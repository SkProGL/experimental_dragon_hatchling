# BDH
OUTPUT (interpolation):
start=3; rule=even:+3,odd:*2; steps=3;
3 -> 6 -> 9 -> 18
============================================================
OUTPUT (interpolation):
start=5; rule=even:+2,odd:+1; steps=4;
5 -> 6 -> 8 -> 10 -> 12
============================================================
OUTPUT (extrapolation):
start=3; rule=even:+3,odd:*2; steps=8;
3 -> 6 -> 9 -> 18 -> 21
============================================================
OUTPUT (extrapolation):
start=4; rule=even:*2,odd:+3; steps=10;
40 -> 80 -> 160 -> 320 -> 640 -> 1280
============================================================
OUTPUT (extrapolation):
start=7; rule=even:+3,odd:*2; steps=7;
7 -> 14 -> 17 -> 34 -> 37
============================================================
OUTPUT (extrapolation):
start=10; rule=even:+6,odd:+5; steps=1;
10 -> 13 -> 26 -> 29 -> 58 -> 61 -> 122 -> 125
============================================================

# Transformer
OUTPUT (interpolation):
start=3; rule=even:+3,odd:*2; steps=3;
 ->  2 -> 5 -> 10
============================================================
OUTPUT (interpolation):
start=5; rule=even:+2,odd:+1; steps=4;
5 -> 6 -> 8 -> 10 -> 12
============================================================
OUTPUT (extrapolation):
start=3; rule=even:+3,odd:*2; steps=8;
 -> 2 -> 5 -> 10 -> 13
============================================================
OUTPUT (extrapolation):
start=4; rule=even:*2,odd:+3; steps=10;
1 -> 4 -> 8 -> 16
============================================================
OUTPUT (extrapolation):
start=7; rule=even:+3,odd:*2; steps=7;
1 -> 2 -> 5 -> 10 -> 13 -> 26
============================================================
OUTPUT (extrapolation):
start=10; rule=even:+6,odd:+5; steps=1;
10 -> 12 -> 14 -> 16 -> 18
============================================================

