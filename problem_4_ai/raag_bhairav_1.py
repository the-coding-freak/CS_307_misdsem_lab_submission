import numpy as np
import random
import math
from dataclasses import dataclass
from typing import List, Tuple
from midiutil import MIDIFile

# ============== 1. RAAG BHAIRAV MUSIC THEORY (with Rhythm) ==============

@dataclass
class RaagBhairav:
    """Defines the musical and rhythmic rules for Raag Bhairav."""
    NOTES = {'Sa': 0, 'Re♭': 1, 'Ga': 4, 'Ma': 5, 'Pa': 7, 'Dha♭': 8, 'Ni': 11, 'Sa\'': 12}
    ALLOWED_NOTES = ['Sa', 'Re♭', 'Ga', 'Ma', 'Pa', 'Dha♭', 'Ni', 'Sa\'']
    VADI, SAMVADI = 'Dha♭', 'Re♭'
    PAKAD_PHRASES = [
        ['Sa', 'Ga', 'Ma', 'Dha♭', 'Pa'], ['Ga', 'Ma', 'Re♭', 'Sa'], ['Pa', 'Dha♭', 'Ma', 'Ga'],
        ['Ma', 'Pa', 'Dha♭', 'Ni', 'Sa\''], ['Sa\'', 'Ni', 'Dha♭', 'Pa', 'Ma']
    ]
    # Rhythmic patterns (durations in beats) that sum to 16 (Teentaal)
    TEENTAAL_PATTERNS = [
        [4, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 2, 1, 1, 2, 4, 4],
        [3, 3, 2, 3, 3, 2]
    ]

    @staticmethod
    def get_midi_pitch(note: str, octave: int = 4) -> int:
        base_pitch = 60  # Middle C (C4)
        if note.endswith('\''):
            octave += 1
            note = note[:-1]
        return base_pitch + RaagBhairav.NOTES[note] + (octave - 4) * 12

# ============== 2. AI FITNESS EVALUATOR (with Rhythm Awareness) ==============

class MelodyEvaluator:
    """Scores a melody's (note, duration) adherence to the rules."""
    def __init__(self):
        self.raag = RaagBhairav()
        self.note_indices = {note: i for i, note in enumerate(self.raag.ALLOWED_NOTES)}

    def evaluate_fitness(self, melody: List[Tuple[str, float]]) -> float:
        """Calculates a single fitness score for the melody."""
        if not melody: return 0.0
        notes = [item[0] for item in melody] # Extract notes for melodic evaluation
        
        score = (
            self.check_grammar(notes) * 30 +
            self.check_phrases(notes) * 25 +
            self.check_emphasis(notes) * 20 +
            self.check_aesthetics(melody) * 25 # Aesthetics now evaluates rhythm
        )
        return max(0, score)

    def check_grammar(self, notes: List[str]) -> float:
        # This function remains largely the same as it evaluates pitch sequence
        score = 1.0
        for note in notes:
            if note not in self.raag.ALLOWED_NOTES: score -= 0.2
        return max(0, score)

    def check_phrases(self, notes: List[str]) -> float:
        score, melody_str = 0.0, ''.join(notes)
        for phrase in self.raag.PAKAD_PHRASES:
            if ''.join(phrase) in melody_str:
                score += 1.0 / len(self.raag.PAKAD_PHRASES)
        return min(1.0, score)

    def check_emphasis(self, notes: List[str]) -> float:
        note_counts = {note: notes.count(note) for note in set(notes)}
        total = len(notes)
        vadi_ratio = note_counts.get(self.raag.VADI, 0) / total
        samvadi_ratio = note_counts.get(self.raag.SAMVADI, 0) / total
        score = 0.0
        if 0.15 <= vadi_ratio <= 0.30: score += 0.5
        if 0.10 <= samvadi_ratio <= 0.25: score += 0.5
        return score
    
    def check_aesthetics(self, melody: List[Tuple[str, float]]) -> float:
        """Evaluates melodic arc and rhythmic variety."""
        notes = [item[0] for item in melody]
        durations = [item[1] for item in melody]
        
        # Melodic arc check
        indices = [self.note_indices.get(n, 0) for n in notes]
        has_arc = 0.2 <= indices.index(max(indices)) / len(indices) <= 0.8 if len(indices) > 0 and max(indices) > 0 else False
        
        # Rhythmic variety check
        unique_durations = len(set(durations))
        rhythmic_variety_score = 1.0 if unique_durations >= 3 else 0.5 if unique_durations == 2 else 0.1
        
        return (0.5 * (1.0 if has_arc else 0.2)) + (0.5 * rhythmic_variety_score)

# ============== 3. GENETIC ALGORITHM (with Rhythm Generation) ==============

class GeneticAlgorithmMelodyGenerator:
    """Evolves melodies with pitch and rhythm."""
    def __init__(self, target_beats=32, pop_size=50, generations=100):
        self.target_beats = target_beats
        self.pop_size, self.gens = pop_size, generations
        self.raag, self.evaluator = RaagBhairav(), MelodyEvaluator()

    def create_individual(self) -> List[Tuple[str, float]]:
        """Creates a melody with notes and durations."""
        melody = []
        current_beats = 0
        while current_beats < self.target_beats:
            rhythm_pattern = random.choice(self.raag.TEENTAAL_PATTERNS)
            for duration in rhythm_pattern:
                if current_beats + duration > self.target_beats: continue
                note = random.choice(self.raag.ALLOWED_NOTES)
                melody.append((note, duration))
                current_beats += duration
        return melody

    def crossover(self, p1, p2):
        """Performs crossover on two melodies."""
        pt = random.randint(1, min(len(p1), len(p2)) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

    def mutate(self, melody):
        """Mutates a melody's notes or rhythm."""
        mutated = melody[:]
        if not mutated: return []
        
        idx = random.randint(0, len(mutated) - 1)
        
        if random.random() < 0.7: # Mutate note
            mutated[idx] = (random.choice(self.raag.ALLOWED_NOTES), mutated[idx][1])
        else: # Mutate rhythm by swapping duration with a neighbor
            if idx < len(mutated) - 1:
                d1 = mutated[idx][1]
                d2 = mutated[idx+1][1]
                mutated[idx] = (mutated[idx][0], d2)
                mutated[idx+1] = (mutated[idx+1][0], d1)
        return mutated

    def evolve(self) -> List[Tuple[str, float]]:
        """Runs the genetic algorithm."""
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        for _ in range(self.gens):
            fitnesses = [self.evaluator.evaluate_fitness(ind) for ind in population]
            new_pop = []
            
            # Elitism
            sorted_pop = [p for _, p in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
            new_pop.extend(sorted_pop[:5])
            
            # Offspring generation
            while len(new_pop) < self.pop_size:
                p1 = random.choices(population, weights=fitnesses, k=1)[0]
                p2 = random.choices(population, weights=fitnesses, k=1)[0]
                c1, c2 = self.crossover(p1, p2)
                new_pop.extend([self.mutate(c1), self.mutate(c2)])
            population = new_pop[:self.pop_size]

        final_fitnesses = [self.evaluator.evaluate_fitness(ind) for ind in population]
        return population[final_fitnesses.index(max(final_fitnesses))]

# ============== 4. MIDI & MAIN EXECUTION ==============

def generate_midi(melody: List[Tuple[str, float]], filename: str, tempo: int = 90):
    """Generates a MIDI file to hear the rhythmic melody."""
    midi = MIDIFile(1)
    track, channel, time, volume = 0, 0, 0, 100
    midi.addTrackName(track, time, "Improvised Raag Bhairav")
    midi.addTempo(track, time, tempo)
    midi.addProgramChange(track, channel, time, 104) # Sitar

    for note, duration in melody:
        pitch = RaagBhairav.get_midi_pitch(note)
        vol = 110 if note in (RaagBhairav.VADI, RaagBhairav.SAMVADI) else 100
        midi.addNote(track, channel, pitch, time, duration, vol)
        time += duration

    with open(filename, "wb") as f:
        midi.writeFile(f)
    print(f"OK Melody saved as: {filename}")

def main():
    """Generates a rhythmic melody and saves it as a MIDI file."""
    ai_composer = GeneticAlgorithmMelodyGenerator(target_beats=32)
    best_melody = ai_composer.evolve()
    
    # Print the melody for review
    sargam_line = " ".join([f"{note.replace('♭', 'b')}({dur})" for note, dur in best_melody])
    print("\nGenerated Rhythmic Melody (Note(beats)):")
    print(sargam_line)
    
    # Generate MIDI file to hear the result
    generate_midi(best_melody, filename="improvised_raag_bhairav.mid")

if __name__ == "__main__":
    main()