# Chord2Vec

Description of files
1)'convert_midi.py' converts a directory of midi files into a text file with chords.
It is used as ``$ python convert_midi.py 'midi_dir'``. By default, the output file is called 'data_as_words'.

2)'convert_words.py' reads the word file (e.g. 'data_as_words' from part 1) and writes two files: 'data_as_ids', and 'vocabulary'. By default, these are placed in a new directory called 'data'. It encodes every word (in this case, chord) as a number, up to VOCAB_SIZE. Any unknown words are converted to the number 0. The file 'vocabulary' has the the first VOCAB_SIZE and subsequent lines are the words in the vocabulary followed by the number of times they appear in 'data_as_words'. It starts with 'UNK' (for unknown), followed by words in decreasing order. The order in which they appear is the number they are encoded by. To perform this step, run ``$ python convert_words.py 'data_as_words'``

3)'train.py' will run the Skip-Gram model using by default the data in the folder 'data'. 

4)'analy.ipyn' is a jupyter notebook which 



```
most common chords are
D : 451636
G : 439790
A : 424825
C : 392893
E : 385333
F : 321387
B : 312544
Bb : 265405
Gb : 257890
Db : 235705
Eb : 235000
Ab : 229604
EG : 155932
CE : 147353
CA : 143931
DF : 138334
DB : 136625
GB : 130326
CEG : 128759
FA : 125667
DGB : 122194
GbA : 119192
DG : 117815
DGbA : 116855
DA : 116160
```
