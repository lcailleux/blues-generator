import glob
import pickle
import numpy as np
from helper import constant
from music21 import converter, instrument, note, chord, midi, stream


class DataHandler:
    def load_dataset(self, args):
        print("[INFO] loading MIDI files...")

        notes, pitches, offsets = self.load_notes_and_pitches(args["instrument"])
        input_notes, output_notes, vocab_length, _ = self.get_neural_network_notes(notes, pitches)

        return input_notes, output_notes, vocab_length

    def get_neural_network_notes(self, notes, pitches):
        note_dict = dict()
        vocab_length = len(pitches)
        notes_number = len(notes)

        for i, current_note in enumerate(pitches):
            note_dict[current_note] = i

        sequence_length = constant.SEQUENCE_LENGTH
        num_training = notes_number - sequence_length
        input_notes = np.zeros((num_training, sequence_length, vocab_length))
        output_notes = np.zeros((num_training, vocab_length))

        for i in range(0, num_training):
            input_sequence = notes[i: i + sequence_length]
            output_note = notes[i + sequence_length]

            for j, current_note in enumerate(input_sequence):
                input_notes[i][j][note_dict[current_note]] = 1

            output_notes[i][note_dict[output_note]] = 1

        return input_notes, output_notes, vocab_length, note_dict

    def load_notes_and_pitches(self, desired_instrument):
        notes = []
        offsets = []
        print("[INFO] If a file does not contain one of the required instruments, it will be skipped.")

        for i, file in enumerate(glob.glob(constant.DATASET_PATH + '/' + constant.DATASET_FORMAT)):
            try:
                midi_song = converter.parse(file)
                partitions = instrument.partitionByInstrument(midi_song)

                if partitions:
                    for partition in partitions:
                        if partition.id == desired_instrument:
                            selected_partition = partition

                            for element in selected_partition.flat.notes:
                                offsets.append(element.offset)
                                if isinstance(element, note.Note):
                                    notes.append(str(element.pitch))
                                elif isinstance(element, chord.Chord):
                                    notes.append('.'.join(str(n) for n in element.normalOrder))
                            print("[INFO] song {} partition loaded".format(file))
                            break
            except (IndexError, midi.MidiException) as e:
                print('[INFO] An error happened during song loading: {}. Skipping...'.format(file))
                print(e)
                continue

            #if len(notes) >= 200:#constant.MAX_NOTES:
            if i > 10:
                break

        with open(constant.NOTE_PATH, 'wb') as filepath:
            pickle.dump(notes, filepath)

        print("[INFO] Done loading MIDI files...")
        pitches = sorted(set(item for item in notes))

        return notes, pitches, offsets

    def save_midi(self, args, output, backward_dict):
        final_notes = []
        for element in output:
            index = list(element).index(1)
            final_notes.append(backward_dict[index])

        offset = 0
        partition = stream.Part()
        output_instrument = instrument.fromString(args["instrument"])

        for pattern in final_notes:
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.offset = offset
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                partition.insert(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                partition.insert(new_note)

            offset += 0.5

        print(final_notes)

        partition.insert(0, output_instrument)

        score = stream.Score()
        score.insert(0, output_instrument)
        score.show('text')
        score.write('midi', fp=constant.NEW_MUSIC)
