# Chord2Vec

Word embedding using music chords instead of words. 
The maining training code is modified from 
https://github.com/chiphuyen/tf-stanford-tutorials/blob/master/examples/04_word2vec_visualize.py

Description of files

1)'convert_midi.py' converts a directory of midi files into a text file with chords.
It is used as ``$ python convert_midi.py 'midi_dir'``. By default, the output file is called 'data_as_words'.

2)'convert_words.py' reads the word file (e.g. 'data_as_words' from part 1) and writes two files: 'data_as_ids', and 'vocabulary'. By default, these are placed in a new directory called 'data'. It encodes every word (in this case, chord) as a number, up to VOCAB_SIZE. Any unknown words are converted to the number 0. The file 'vocabulary' has the the first VOCAB_SIZE and subsequent lines are the words in the vocabulary followed by the number of times they appear in 'data_as_words'. It starts with 'UNK' (for unknown), followed by words in decreasing order. The order in which they appear is the number they are encoded by. To perform this step, run ``$ python convert_words.py 'data_as_words'``

3)'train.py' will run the Skip-Gram model using by default the data in the folder 'data'. 

4)'analyze.ipyn' is a jupyter notebook which shows some analysis. It also produces a random midi file by skipping around nearest neighbours.

I used a vocabulary size of 500, and trained the model on about 11 million chords (from 150 MB of midi files) for about an hour using an embedding dimension of 32 and a skip window of 6
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

```
closest to D : ['DBb', 'Eb', 'DG', 'DGb', 'Db']
closest to G : ['B', 'E', 'GB', 'CG', 'DG']
closest to A : ['DbA', 'Db', 'EA', 'DA', 'B']
closest to C : ['CG', 'CBb', 'CE', 'CD', 'CF']
closest to E : ['G', 'Db', 'EG', 'EB', 'C']
closest to F : ['FBb', 'FAb', 'DF', 'Eb', 'Bb']
closest to B : ['G', 'DB', 'Ab', 'DbB', 'Db']
closest to Bb : ['EbBb', 'Ab', 'F', 'Eb', 'EbG']
closest to Gb : ['DGb', 'GbB', 'G', 'DGbA', 'CGb']
closest to Db : ['DbA', 'E', 'DbG', 'B', 'A']
closest to Eb : ['EbG', 'F', 'Bb', 'EbA', 'CEb']
closest to Ab : ['DAb', 'Bb', 'CAb', 'F', 'AbB']
closest to EG : ['EGB', 'CEG', 'EB', 'CE', 'E']
closest to CE : ['CEG', 'C', 'CG', 'CEF', 'EG']
closest to CA : ['C', 'CFA', 'CGbA', 'CF', 'CGA']
closest to DF : ['F', 'FBb', 'DFBb', 'Eb', 'Bb']
closest to DB : ['B', 'DGB', 'DFB', 'DEB', 'GB']
closest to GB : ['EGB', 'G', 'B', 'DGB', 'DB']
closest to CEG : ['CE', 'CG', 'EG', 'C', 'CEBb']
closest to FA : ['DFA', 'F', 'EFA', 'DF', 'CFA']
closest to DGB : ['DB', 'GB', 'DEGB', 'DG', 'G']
closest to GbA : ['DGbA', 'CGbA', 'Gb', 'DGb', 'GbB']
closest to DG : ['G', 'DGbG', 'CDG', 'DBb', 'DGB']
closest to DGbA : ['DGb', 'DA', 'GbA', 'DEGbA', 'Gb']
closest to DA : ['DGbA', 'A', 'CDA', 'DFA', 'DAbA']
closest to DGb : ['Gb', 'DGbA', 'DGbG', 'DGbB', 'GbB']
closest to CG : ['C', 'CEG', 'CE', 'G', 'CDG']
closest to GBb : ['DGBb', 'EbGBb', 'Bb', 'DBb', 'FGBb']
closest to DBb : ['DGBb', 'Bb', 'GBb', 'D', 'FBb']
closest to EA : ['A', 'DbA', 'E', 'CEA', 'EAB']
closest to CFA : ['CF', 'F', 'C', 'CA', 'FBb']
closest to DbE : ['DbDE', 'DbEB', 'Db', 'DbEG', 'E']
closest to DbEA : ['EA', 'DbA', 'DbEGA', 'DbDEA', 'DbEFA']
closest to DbA : ['Db', 'A', 'EA', 'DbGbA', 'DbG']
closest to CEb : ['Eb', 'CEbF', 'C', 'Bb', 'CEbG']
closest to CF : ['CFA', 'F', 'C', 'CFBb', 'CBb']
closest to EbG : ['Eb', 'EbGBb', 'EbBb', 'Bb', 'CEbG']
closest to DFBb : ['DFAbBb', 'FBb', 'Bb', 'DF', 'DFGBb']
closest to EB : ['EGB', 'EAbB', 'EG', 'B', 'E']
closest to DFA : ['FA', 'EFA', 'DA', 'DEFA', 'DFABb']
closest to AbB : ['DAbB', 'FAbB', 'Ab', 'EAbB', 'CAbB']
closest to EAb : ['EAbB', 'CEAb', 'Ab', 'DEAb', 'DbEAb']
closest to FAb : ['F', 'CFAb', 'CAb', 'Ab', 'EbFAb']
closest to CEA : ['CE', 'EA', 'CA', 'CEGbA', 'CEB']
closest to FBb : ['F', 'Bb', 'CFBb', 'DF', 'DFBb']
closest to EAbB : ['EAb', 'EB', 'AbB', 'EbEAbB', 'EGbAbB']
closest to EbGBb : ['EbG', 'GBb', 'EbBb', 'EbGAbBb', 'Bb']
closest to CAb : ['Ab', 'CFAb', 'EbAb', 'FAb', 'Bb']
closest to DGBb : ['GBb', 'DBb', 'DGABb', 'DG', 'DFAbBb']
```
