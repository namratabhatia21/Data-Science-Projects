def has_consecutive_caps(text):
    """ Function to check if four consecutive characters in a tweet are upper case"""
    for i in range(len(text) - 3):
        if text[i:i+4].isupper():
            return True
    return False

def plot_top_n_words(reviews, N=20):
    """
    Plots the top N words by frequency in the given reviews.
    
    :param reviews: Iterable of text reviews
    :param N: Number of top words to visualize
    """
    # Step 1: Tokenize the text and convert to lowercase
    all_words = [word for review in reviews for word in word_tokenize(str(review).lower())]
    
    # Step 2: Count the frequency of each word
    word_counts = Counter(all_words)
    
    # Step 3: Select the top N words
    top_words = word_counts.most_common(N)
    
    # Step 4: Visualize these top words and their counts
    words, counts = zip(*top_words)  # Unzip the tuples into two lists
    
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {N} Words in Reviews by Count')
    plt.xticks(rotation=45)  # Rotate the word labels for better readability
    plt.show()

# Function to get top N bi-grams
def get_top_n_bigrams(corpus, n=None):
    vec = [bigram for l in corpus for bigram in bigrams(word_tokenize(str(l)))]
    counts = Counter(vec)
    return counts.most_common(n)

# Function to get top N tri-grams
def get_top_n_trigrams(corpus, n=None):
    vec = [trigram for l in corpus for trigram in trigrams(word_tokenize(str(l)))]
    counts = Counter(vec)
    return counts.most_common(n)

def spacy_cleaner(text):
    """
    Preprocesses text using Spacy, including handling contractions, removing special characters, and stop words.

    Args:
    text (str): Input text to be cleaned.

    Returns:
    str: Cleaned and preprocessed text.
    """
    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)
    
    # Replace curly apostrophe with straight apostrophe
    apostrophe_handled = re.sub("’", "'", decoded)

    # Expand contractions
    expanded = contractions.fix(apostrophe_handled)
    
    # Tokenize the text using Spacy
    parsed = nlp(expanded)
    
    final_tokens = []
    # Process each token
    for t in parsed:
        # Skip punctuation, whitespace, numbers, URLs, mentions, and stop words
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@') or t.is_stop or \
                'covid' in t.lower_ or 'coronavirus' in t.lower_ or 'covid19' in t.lower_ or 'corona' in t.lower_ or 'amp' in t.lower_:
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                # Remove non-alphabetic characters and lemmatize
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    
    # Join the final tokens into a string
    joined = ' '.join(final_tokens)
    
    # Perform spell correction (repeating characters)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    
    return spell_corrected

def split_hashtag_to_words_all_possibilities(hashtag):
	all_possibilities = []
	
	split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag)+1))]
	possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]
	
	for split_pos in possible_split_positions:
		split_words = []
		word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]
		
		if word_2 in word_dictionary:
			split_words.append(word_1)
			split_words.append(word_2)
			all_possibilities.append(split_words)

			another_round = split_hashtag_to_words_all_possibilities(word_2)
				
			if len(another_round) > 0:
				all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
		else:
			another_round = split_hashtag_to_words_all_possibilities(word_2)
			
			if len(another_round) > 0:
				all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
	
	return all_possibilities

def lr_cv(splits, X, Y, pipeline, average_method):
    """
    Perform cross-validation for logistic regression model.

    Args:
    splits (int): Number of splits for cross-validation.
    X (array-like): Feature matrix.
    Y (array-like): Target labels.
    pipeline: Pipeline for logistic regression.
    average_method (str): Method for averaging precision, recall, and F1 scores.

    Returns:
    None
    """
    # Initialize lists to store evaluation metrics
    accuracy = []
    precision = []
    recall = []
    f1 = []

    # Define the StratifiedKFold object
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)

    # Iterate over the folds
    for train_indices, test_indices in kfold.split(X, Y):

        # Access the rows of X and Y corresponding to the train indices
        X_train, Y_train = X.iloc[train_indices], Y.iloc[train_indices]

        # Fit the pipeline on the training data
        lr_fit = pipeline.fit(X_train, Y_train)
        
        # Make predictions on the test data
        prediction = lr_fit.predict(X.iloc[test_indices])
        
        # Calculate accuracy
        scores = lr_fit.score(X.iloc[test_indices], Y.iloc[test_indices])
        accuracy.append(scores * 100)
        
        # Calculate precision, recall, and F1 score
        precision.append(precision_score(Y.iloc[test_indices], prediction, average=average_method) * 100)
        recall.append(recall_score(Y.iloc[test_indices], prediction, average=average_method) * 100)
        f1.append(f1_score(Y.iloc[test_indices], prediction, average=average_method) * 100)
        
        # Print precision, recall, and F1 score for each class
        print('              Positive, Negative, Neutral, Extremely Positive, Extremely Negative ')
        print('precision:', precision_score(Y.iloc[test_indices], prediction, average=None))
        print('recall:   ', recall_score(Y.iloc[test_indices], prediction, average=None))
        print('f1 score: ', f1_score(Y.iloc[test_indices], prediction, average=None))
        print('-' * 50)

    # Print average evaluation metrics
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))

# Synonym Replacement
def augment_synonym(sentence, num_augments=1):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_sentences = [aug.augment(sentence) for _ in range(num_augments)]
    return augmented_sentences

# Random Insertion
def augment_insertion(sentence, num_augments=1):
    words = sentence.split()
    augmented_sentences = []
    for _ in range(num_augments):
        insert_word = random.choice(words)
        insert_position = random.randint(0, len(words))
        new_words = words[:insert_position] + [insert_word] + words[insert_position:]
        augmented_sentence = ' '.join(new_words)
        augmented_sentences.append(augmented_sentence)
    return augmented_sentences

# Simulated Shuffle
def augment_shuffle(sentence, num_augments=1):
    augmented_sentences = []
    for _ in range(num_augments):
        words = sentence.split()  # Ensure a new list is created for each augmentation
        random.shuffle(words)
        augmented_sentences.append(' '.join(words))
    return augmented_sentences

# Apply augmentations to a list of sentences
def augment_sentences(sentences, augment_fn, total_augments_needed):
    augmented_data = []
    for sentence in sentences:
        if total_augments_needed <= 0:
            break  # Stop if we have reached the required number of augmentations
        augmented_data.extend(augment_fn(sentence, 1))  # Augment each sentence once at a time
        total_augments_needed -= 1
    return augmented_data

def evaluate_model(X_test, y_test, model):
    labels_index = {'Extremely Positive': 4, 'Positive': 3, 'Neutral': 2, 'Negative': 1, 'Extremely Negative': 0}
    # Define the class labels
    labels = list(labels_index.keys())

    # Make predictions
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

    # Classification report
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()