import mido
import numpy as np

# takes boolean array of size 12 and converts it to a string with the note names
# separated by '-'
def name_notes(chord, key_signature='C'):
    # (not using key_signature)
    #if key_signature in {'C', 'Db', 'Eb', 'F', 'Ab', 'Bb'}:    
    names = np.array(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])
    return ''.join(names[np.where(chord)])


# reads the chords from a track
# returns list of chords
# each chord is a string
def read_chords_old(track):
    chords = []
    
    buffer = []
    chord = np.zeros([12], dtype=bool)
    #key_signature = 'C'
    for msg in track:

            # keep track of key signature (haven't used this)
            #if msg.is_meta and msg.type == 'key_signature':
            #    print('Key Signature', msg.key)
            #    key_signature = msg.key

            # process the buffer 
            if msg.time > 0:
                # if the buffer is not empty
                if buffer:
                    for bmsg in buffer:
                        note = bmsg.note % 12
                        chord[note] = 1 if bmsg.velocity > 0 else 0

                    # append the chord as a string of notes if it is not empty
                    if np.any(chord): 
                        chord_notes = name_notes(chord)
                        # print('Chord', chord_notes)
                        chords.append(chord_notes)

                # reset buffer, starting with current message if it is a note message
                buffer = [msg] if msg.type == 'note_on' else []

            # add note messages to the buffer if time == 0
            elif msg.type == 'note_on':
                buffer.append(msg)
    
    return chords

# reads the chords from a track
# returns list of chords
# each chord is a string
# This version also takes 'note_off' messages into account
def read_chords(track):
    chords = []
    
    buffer = []
    chord = np.zeros([12], dtype=bool)
    #key_signature = 'C'
    for msg in track:

            # keep track of key signature (haven't used this)
            #if msg.is_meta and msg.type == 'key_signature':
            #    print('Key Signature', msg.key)
            #    key_signature = msg.key

            # process the buffer 
            if msg.time > 0:
                # if the buffer is not empty
                if buffer:
                    for bmsg in buffer:
                        
                        note = bmsg.note % 12
                        if bmsg.type == 'note_on':
                            chord[note] = 1 if bmsg.velocity > 0 else 0
                        
                        elif bmsg.type == 'note_off':
                            #print(bmsg)
                            chord[note] = 0

                    # append the chord as a string of notes if it is not empty
                    if np.any(chord): 
                        chord_notes = name_notes(chord)
                        # print('Chord', chord_notes)
                        chords.append(chord_notes)

                # reset buffer, starting with current message if it is a note message
                buffer = [msg] if msg.type in {'note_on', 'note_off'} else []

            # else time is 0, add messages if they are note types
            elif msg.type in {'note_on', 'note_off'}:
                buffer.append(msg)
    
    return chords

# merges list of tracks into one track
# returns mido Track
def merge_tracks(tracks):
    tmp = []
    
    # I don't use tracknum for anything at the moment
    # but maybe later if I want each track to be it's own channel or something
    for tracknum, track in enumerate(tracks):
        global_t = 0 # absolute time to be used to sort msg order
        buffer = [] # buffer for messages from this track
        for msg in track:
            global_t += msg.time
            buffer.append((msg, global_t, tracknum)) # append message, absolute time, and tracknum
        tmp += buffer

    # sort the triples in tmp by their absolute time, which is the second component
    tmp.sort(key=lambda triple : triple[1])

    # now we will unpack these messages and recompute their time    
    track0 = []
    global_t = 0
    for msg, t, _ in tmp:
        # new message time, msg copy is recommended instead of modifying attribute
        new_msg = msg.copy(time= (t - global_t))
        
        track0.append(new_msg)
        global_t = t # update absolute time

    return mido.MidiTrack(track0)

# converts midi file to type 0 by using the merge_tracks function
# input and output are mido MidiFile types
def convert(old_mid):
    
    if old_mid.type == 0:
        print('Already Type 0')
    
    else:
        new_mid = mido.MidiFile(type=0)
        new_mid.tracks.append(merge_tracks(old_mid.tracks))
        #new_mid.save(output_file)
    
    return new_mid

def read_midi_file(file_path):
    try:
        mid = mido.MidiFile(file_path)
        print('Reading', file_path)
    except:
        print('Error Reading', file_path)
    else:
        if mid.type != 0:
            try:
                mid = convert(mid)
            except:
                print('Error Converting File')
                return []


        return read_chords(mid.tracks[0])




