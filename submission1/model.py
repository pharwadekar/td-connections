import os
from groq import Groq
from dotenv import load_dotenv
import ast
import numpy as np
from itertools import chain
import gensim.downloader as api

model = api.load("word2vec-google-news-300")  # Loading Google's Word2Vec model (300-dimensional vectors)

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    _______________________________________________________
    Parameters:
    words - 1D Array with 16 shuffled words
    strikes - Integer with number of strikes
    isOneAway - Boolean if your previous guess is one word away from the correct answer
    correctGroups - 2D Array with groups previously guessed correctly
    previousGuesses - 2D Array with previous guesses
    error - String with error message (0 if no error)

    Returns:
    guess - 1D Array with 4 words
    endTurn - Boolean if you want to end the puzzle
    _______________________________________________________
    """

    # Your Code here

    def create_env_file(api_key):
        with open('.env', 'w') as f:
            f.write(f'GROQ_API_KEY="{"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"}"')

    load_dotenv()

    def initialize_groq():
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            api_key = input("Please enter your Groq API key: ")
            create_env_file(api_key)
            load_dotenv()
        return Groq(api_key=api_key)

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def find_weakest_word(words, word_vectors):
        avg_similarities = []
        for i, word_vector in enumerate(word_vectors):
            similarities = []
            for j, other_vector in enumerate(word_vectors):
                if i != j:
                    similarity = cosine_similarity(word_vectors[i], word_vectors[j])
                    similarities.append(similarity)
            avg_similarity = np.mean(similarities)
            avg_similarities.append((words[i], avg_similarity))

        avg_similarities.sort(key=lambda x: x[1])
        return avg_similarities[0][0]

    def replace_weakest_with_best(word_list, current_guess, word_vectors):
        remaining_words = [word for word in word_list if word not in current_guess]
        best_word = None
        highest_similarity = -1

        for word in remaining_words:
            word_vector = word_vectors[word]
            avg_similarity = 0
            for guess_word in current_guess:
                avg_similarity += cosine_similarity(word_vectors[guess_word], word_vector)
            avg_similarity /= len(current_guess)

            if avg_similarity > highest_similarity:
                highest_similarity = avg_similarity
                best_word = word

        weakest_word = find_weakest_word(current_guess, [word_vectors[word] for word in current_guess])
        current_guess[current_guess.index(weakest_word)] = best_word
        #print(f"Replaced '{weakest_word}' with '{best_word}'")

        return current_guess

    flattened_correctGroups = list(chain.from_iterable(correctGroups))

    client = initialize_groq()
    failed_one_away_attempts = set()  # Track failed "one away" attempts

    wordList = [word for word in words if word not in flattened_correctGroups]

    # Create word vectors using the Word2Vec model (300-dimensional vectors)
    word_vectors = {}

    for word in wordList:
        try:
            word_vectors[word] = model[word]  # 300-dimensional vector for each word
        except KeyError:
            # If the word is not found in the model, assign a random vector (fallback)
            word_vectors[word] = np.random.rand(300)  # Use a random 300-dimensional vector

    remaining_attempts = 15  # Limit total attempts to prevent infinite loops

    while remaining_attempts > 0:
        remaining_attempts -= 1
        prompt = f"""
        Take the words: {wordList} THIS IS THE CURRENT WORD BANK
        Previous Guesses that were WRONG: {previousGuesses} DO NOT REDO THESE PAST GUESSES
        
        ### Game Rules:
        - Each word belongs to only one of the four groups of four.
        - To play, players select four words they believe belong to a specific category.
        - Correctly identifying a full set of four words allows the player to continue.

        The game's objective is to categorize 4 words in {wordList} accurately, emphasizing strategic selection based on similarity. High-quality sets are those with well-defined, distinct categories to support player success.
        1. Does not match any previous incorrect guesses in {previousGuesses}.
        2. Contains only the words in {wordList}.
        3. Forms a very clear, simple, and easily recognizable category or theme that most people would immediately understand.

        The output should look like this: ["word1", "word2", "word3", "word4"]
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=1,
            max_tokens=1024
        )

        response = chat_completion.choices[0].message.content
        response = response[response.index("["): response.index("]")+1]
        #print(f"Response from model: {response}")

        guessed_group = ast.literal_eval(response)

        # Skip if this exact group has been guessed before
        if any(set(guessed_group) == set(prev_guess) for prev_guess in previousGuesses):
            #print(f"Skipping repeated guess: {guessed_group}")
            continue

        # Validate that the guess only contains available words
        if not all(word in wordList for word in guessed_group):
            #print("Invalid guess: contains words not in the available list")
            continue

        if isOneAway:
            prompt = f"""
            Previous Guess: {previousGuesses[-1]}
            Remaining Words: {wordList}

            REPLACE one WORD from the Previous guess, so that the grouping is more cohesive

            The output should look like this: ["word1", "word2", "word3", "replacementword"]

            Game Rules:
            - Connections is a word categorization game with 16 words.
            - Each word belongs to only one of four categories.
            - Correct guesses are added to "correctGroups" and "previousGuesses".
            - Provide the best categorization of these words into four sets.
            Parameters:
            - words: {wordList}
            - previousGuesses: {previousGuesses}
            """

            current_attempt = frozenset(previousGuesses[-1])

            if current_attempt in failed_one_away_attempts:
                #print("This combination has already failed in a one-away attempt")
                continue

            guessed_group = replace_weakest_with_best(wordList, current_attempt.copy(), word_vectors)

        # Game end conditions
        guess = guessed_group
        endTurn = False  # Return False to continue the puzzle

        return guess, endTurn