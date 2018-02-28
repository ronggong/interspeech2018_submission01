# Phonetic segmentation demo

This is the demo code to run the duration-informed phoneme segmentation
algorithm for a singing phrase example.

The example .wav, syllable and phoneme durations are included in the `.\temp\`
folder. The `syllable_durations.npy` is a numpy array of the syllable durations
obtained by annotating the teacher's demostrative singing. `phoneme_durations_grouped_by_syllables.npy`
is a array containing sub-arrays. One array is a syllable unit which containing the phoneme durations.
 
To run the demo, you need to first install the `requirements.txt`.

Then run `distribute_proposed_method_jupyter.ipynb` to see the detected syllable onset times and
phoneme onset times.