import re
from collections import Counter
from typing import List, Set, Tuple
import time

class NorvigSpellCorrector:
    def __init__(self, dictionary_path: str):
        """Initialize the spell corrector with a dictionary corpus."""
        self.WORDS = Counter(self._words(open(dictionary_path, encoding='utf8').read()))
        self.N = sum(self.WORDS.values())

    def _words(self, text: str) -> List[str]:
        """Extract words from text."""
        return re.findall(r'\w+', text.lower())

    def probability(self, word: str) -> float:
        """Probability of a word based on its frequency in the corpus."""
        return self.WORDS[word] / self.N if word in self.WORDS else 0

    def correction(self, word: str) -> str:
        """Most probable spelling correction for a word."""
        return max(self.candidates(word), key=self.probability)

    def candidates(self, word: str) -> Set[str]:
        """Generate possible spelling corrections for a word."""
        return (self.known([word]) or 
                self.known(self.edits1(word)) or 
                self.known(self.edits2(word)) or 
                {word})

    def known(self, words: List[str]) -> Set[str]:
        """Filter the list of words to those present in the dictionary."""
        return {w for w in words if w in self.WORDS}

    def edits1(self, word: str) -> Set[str]:
        """All edits that are one edit away from the word."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:]          for L, R in splits if R for c in letters]
        inserts = [L + c + R              for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> Set[str]:
        """All edits that are two edits away from the word."""
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def unit_tests(self) -> str:
        """Run unit tests to validate spell corrector logic."""
        assert self.correction('speling') == 'spelling'
        assert self.correction('korrectud') == 'corrected'
        assert self.correction('bycycle') == 'bicycle'
        assert self.correction('inconvient') == 'inconvenient'
        assert self.correction('arrainged') == 'arranged'
        assert self.correction('peotry') == 'poetry'
        assert self.correction('peotryy') == 'poetry'
        assert self.correction('word') == 'word'
        assert self.correction('quintessential') == 'quintessential'
        assert self._words('This is a TEST.') == ['this', 'is', 'a', 'test']
        assert Counter(self._words('This is a test. 123; A TEST this is.')) == (
               Counter({'123': 1, 'a': 2, 'is': 2, 'test': 2, 'this': 2}))
        return 'All unit tests passed.'

    def test_set(self, lines: List[str]) -> List[Tuple[str, str]]:
        """Parse test lines into (correct, incorrect) pairs."""
        return [(right, wrong)
                for line in lines if ':' in line
                for (right, wrongs) in [line.strip().split(':')]
                for wrong in wrongs.strip().split()]

    def spelltest(self, test_data: List[Tuple[str, str]], verbose=False):
        """Run tests and evaluate spell corrector accuracy."""
        start = time.perf_counter()
        correct = total = unknown = 0

        for right, wrong in test_data:
            total += 1
            predicted = self.correction(wrong)
            if predicted == right:
                correct += 1
            else:
                if right not in self.WORDS:
                    unknown += 1
                if verbose:
                    print(f'correction({wrong}) => {predicted}; expected {right}')

        elapsed = time.perf_counter() - start
        print(f'{correct/total:.0%} correct of {total} ({unknown/total:.0%} unknown), '
              f'{total / elapsed:.0f} words/sec')

