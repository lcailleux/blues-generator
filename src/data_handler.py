import pickle
import numpy as np
from keras.utils import np_utils
from helper import constant
from helper import utils
from music21 import converter, instrument, note, chord, midi, stream


class DataHandler:
    def load_dataset(self, args):
        print("[INFO] loading MIDI file...")

        notes, offsets = self.load_partition(args)
        vocab_length = len(set(notes))
        network_input, network_output = self.prepare_sequences(notes, args['sequence_length'], vocab_length)

        return network_input, network_output, vocab_length

    def prepare_sequences(self, notes, sequence_length, vocab_length):
        if not sequence_length or sequence_length > constant.SEQUENCE_LENGTH:
            sequence_length = constant.SEQUENCE_LENGTH

        print(sequence_length)
        pitches = sorted(set(item for item in notes))
        note_to_int = dict((current_note, number) for number, current_note in enumerate(pitches))

        network_input = []
        network_output = []

        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        network_input = network_input / float(vocab_length)
        network_output = np_utils.to_categorical(network_output)

        return network_input, network_output

    def load_partition(self, args):
        notes = []
        offsets = []

        try:
            with open(args["file"], 'rb') as file:
                midi_song = converter.parse(file.read())
                instrument_name, partition = self.choose_instrument_partition(midi_song)

                try:
                    metronome_mark_boundaries = partition.metronomeMarkBoundaries()[0]
                    metronome_mark = utils.get_latest_tuple_element(metronome_mark_boundaries)
                except IndexError:
                    metronome_mark = ''

                partition_info = {
                    "instrument_name": instrument_name,
                    'key': self.get_partition_element_by_class(partition, 'KeySignature'),
                    'time_signature': self.get_partition_element_by_class(partition, 'TimeSignature'),
                    'metronome_mark': metronome_mark,
                    'sequence_length': args['sequence_length']
                }

                for element in partition.flat.notes:
                    if isinstance(element, note.Note):
                        offsets.append(element.offset)
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        offsets.append(element.offset)
                        notes.append('.'.join(str(n) for n in element.normalOrder))

                if not notes:
                    raise ValueError("There is not enough data in this partition. Please pick another one.")

                self.save_partition_info_and_notes(partition_info, notes)
                print("[INFO] song {} partition loaded".format(file.name))
                print("[INFO] Done loading MIDI file...")
        except (IndexError, ValueError, midi.MidiException) as e:
            print('[ERROR] An error happened during song loading: {}. Quitting...'.format(file.name))
            print(e)
            exit()

        return notes, offsets

    def get_partition_element_by_class(self, partition, class_to_find):
        element = partition.getElementsByClass(class_to_find)
        for element in element:
            return element
        return ''

    def choose_instrument_partition(self, midi_song):
        instrument_name = None
        song_instruments = []
        partitions = instrument.partitionByInstrument(midi_song)
        input_question = "Please select one of the instrument for music generation:"

        if not partitions:
            raise ValueError('partitions not found.')

        for partition in partitions:
            if partition.id:
                if len(partition.flat.notes) > 1:
                    song_instruments.append(partition.id)
                    input_question += "\n\t" + partition.id

        input_question += '\n:'
        while instrument_name not in song_instruments:
            instrument_name = input(input_question)

        return instrument_name, partitions.getElementById(instrument_name)

    def save_partition_info_and_notes(self, partition_info, notes):
        with open(constant.NOTE_PATH, 'wb') as note_path:
            pickle.dump(notes, note_path)

        with open(constant.PARTITION_INFO_PATH, 'wb') as partition_info_path:
            pickle.dump(partition_info, partition_info_path)

    def save_midi(self, partition_info, prediction_output):
        offset = 0

        partition = stream.Part()
        elements_to_insert = ['key', 'time_signature', 'metronome_mark']
        for element in elements_to_insert:
            if partition_info[element]:
                partition.insert(partition_info[element])

        partition.insert(instrument.fromString(partition_info['instrument_name']))

        for pattern in prediction_output:
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    notes.append(new_note)
                partition.insert(offset, chord.Chord(notes))
            else:
                partition.insert(offset, note.Note(pattern))

            offset += 0.5

        midi_stream = stream.Stream()
        midi_stream.insert(0, partition)
        midi_stream.show("text")

        midi_stream.write('midi', fp=constant.NEW_MUSIC)
