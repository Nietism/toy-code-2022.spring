Repetition (self-amplification effect) is more complicated. The solutions in the homework handout are unconsidered.



To reduce repetition:

+ Don't repeat *n*-grams (*n*-gram penalty). 

  For example, the ***no_repeat_ngram_size*** parameter in Huggingface transformers: https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/generation/configuration_utils.py#L38

+ Use a different training objective.

+ Use a different decoding objective.



More information: 

[cs224n-2023-lecture-10-NLG]: http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture10-nlg.pdf
