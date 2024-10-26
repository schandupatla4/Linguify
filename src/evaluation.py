from nltk.translate.gleu_score import sentence_gleu

def evaluate_gleu(corrected_sentences, test_data):
    gleu_scores = []
    for corrected_sentence, references in zip(corrected_sentences, test_data["corrections"]):
        prediction_tokens = corrected_sentence.split()
        reference_tokens = [ref.split() for ref in references]
        gleu_score = sentence_gleu(reference_tokens, prediction_tokens)
        gleu_scores.append(gleu_score)

    avg_gleu_score = sum(gleu_scores) / len(gleu_scores)
    print(f"\nAverage GLEU Score: {avg_gleu_score:.4f}")
    return avg_gleu_score
