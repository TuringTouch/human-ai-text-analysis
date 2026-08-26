"""
TOPIC MODELING TEMPLATE FOR HUMAN–AI TEXT RESEARCH
===================================================

PURPOSE
-------
This script provides a reusable and extensively documented template for
topic modeling in computational text analysis.

The template is designed for researchers working with textual data such as:

    - Reddit posts or comments
    - online discussions
    - interview transcripts
    - open-ended survey responses
    - reviews
    - organizational documents
    - other textual corpora

IMPORTANT:
This repository contains CODE and DOCUMENTATION only.

Keep your actual research dataset on your local machine.


METHODS INCLUDED
----------------

This template introduces three complementary topic-modeling approaches:

1. LDA
   Latent Dirichlet Allocation

2. NMF
   Non-negative Matrix Factorization

3. BERTopic
   Embedding-based topic modeling


WHY THREE METHODS?
------------------

Different topic-modeling approaches make different assumptions.

LDA:
    Classical probabilistic topic model.
    Strong choice for transparent exploratory analysis.

NMF:
    Matrix-factorization approach based on TF-IDF.
    Useful as an alternative / robustness approach.

BERTopic:
    Embedding-based approach.
    Particularly useful when semantic similarity matters.

The purpose is NOT to run every method automatically and then choose
whichever produces the "prettiest" topics.

Instead, researchers should compare:

    - statistical diagnostics
    - topic coherence
    - topic diversity
    - topic stability
    - human interpretability
    - theoretical relevance

"""


# =============================================================================
# 0. IMPORT PACKAGES
# =============================================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================
#
# THIS IS THE MOST IMPORTANT SECTION FOR USERS.
#
# Users should normally only need to modify the values in this section.
#
# Everything below is designed to be reusable.
# =============================================================================


# -----------------------------------------------------------------------------
# CHANGE THIS:
# -----------------------------------------------------------------------------
# Path to your LOCAL research dataset.
#
# Example:
#
# DATA_PATH = "/Users/yourname/Desktop/my_dataset.csv"
#
# IMPORTANT:
# Do not upload your real dataset to GitHub if it contains restricted
# user-generated content or other data that cannot be redistributed.
#
# The dataset should remain on your local computer.
# -----------------------------------------------------------------------------

DATA_PATH = "YOUR_LOCAL_DATA_PATH.csv"


# -----------------------------------------------------------------------------
# CHANGE THIS:
# -----------------------------------------------------------------------------
# Name of the column containing the textual material.
#
# Examples:
#
# TEXT_COLUMN = "text"
# TEXT_COLUMN = "body"
# TEXT_COLUMN = "comment"
# TEXT_COLUMN = "post_text"
# -----------------------------------------------------------------------------

TEXT_COLUMN = "text"


# -----------------------------------------------------------------------------
# OPTIONAL:
# -----------------------------------------------------------------------------
# Name of a grouping variable.
#
# This is useful if you want to compare topic prevalence across groups.
#
# Example:
#
# GROUP_COLUMN = "interaction_type"
#
# where the values might be:
#
#     scrutiny
#     acceptance
#
# If you do not have a grouping variable, use:
#
# GROUP_COLUMN = None
# -----------------------------------------------------------------------------

GROUP_COLUMN = None


# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY
# -----------------------------------------------------------------------------
# Results will be saved locally in this folder.
#
# This folder should generally NOT be uploaded to GitHub if it contains
# outputs derived from restricted research data.
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("topic_modeling_results")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. GENERAL ANALYSIS PARAMETERS
# =============================================================================


# -----------------------------------------------------------------------------
# NUMBER OF TOPICS
# -----------------------------------------------------------------------------
#
# IMPORTANT:
#
# Do not assume that one number of topics is automatically correct.
#
# A useful exploratory range might be:
#
#     3, 5, 7, 10, 15
#
# You can later compare models using:
#
#     - coherence
#     - topic diversity
#     - stability
#     - interpretability
#     - theoretical usefulness
#
# N_TOPICS below is simply the number used for the main model.
# -----------------------------------------------------------------------------

N_TOPICS = 7


# -----------------------------------------------------------------------------
# NUMBER OF TOP WORDS PER TOPIC
# -----------------------------------------------------------------------------

N_TOP_WORDS = 20


# -----------------------------------------------------------------------------
# RANDOM SEED
# -----------------------------------------------------------------------------
#
# Topic models can contain stochastic elements.
#
# Fixing the random seed makes results more reproducible.
# -----------------------------------------------------------------------------

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# MINIMUM DOCUMENT FREQUENCY
# -----------------------------------------------------------------------------
#
# Ignore words appearing in fewer than this number of documents.
#
# Example:
#
#     MIN_DF = 5
#
# means that a term must occur in at least five documents.
#
# For very small datasets, use a smaller value.
# For very large datasets, a larger value may be appropriate.
# -----------------------------------------------------------------------------

MIN_DF = 5


# -----------------------------------------------------------------------------
# MAXIMUM DOCUMENT FREQUENCY
# -----------------------------------------------------------------------------
#
# Ignore terms appearing in more than this proportion of documents.
#
# Example:
#
#     MAX_DF = 0.95
#
# means that terms appearing in more than 95% of documents are ignored.
# -----------------------------------------------------------------------------

MAX_DF = 0.95


# -----------------------------------------------------------------------------
# N-GRAM RANGE
# -----------------------------------------------------------------------------
#
# (1, 1)
#     unigrams only
#
# (1, 2)
#     unigrams + bigrams
#
# (1, 3)
#     unigrams + bigrams + trigrams
#
# For many social-media corpora, (1, 2) is a useful starting point.
# -----------------------------------------------------------------------------

NGRAM_RANGE = (1, 2)


# =============================================================================
# 3. LOAD DATA
# =============================================================================

def load_data(path):
    """
    Load the research dataset.

    This template assumes CSV input.

    If your dataset is stored in another format, modify this function.

    Examples:

        pd.read_excel(...)
        pd.read_parquet(...)
        pd.read_json(...)

    IMPORTANT:
    The data remain local. This function does not upload anything anywhere.
    """

    print("\n" + "=" * 70)
    print("STEP 1 — LOADING DATA")
    print("=" * 70)

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            "\nDataset not found.\n\n"
            f"Current path:\n{path}\n\n"
            "Please change DATA_PATH in the USER CONFIGURATION section."
        )

    df = pd.read_csv(path)

    print(
        f"Number of rows: {len(df):,}"
    )

    print(
        f"Columns available:\n{list(df.columns)}"
    )

    return df


# =============================================================================
# 4. VALIDATE DATA
# =============================================================================

def validate_data(df):
    """
    Check whether the expected text column exists.

    This step intentionally happens before preprocessing.

    It is good research practice to fail early when the expected data
    structure is not present.
    """

    print("\n" + "=" * 70)
    print("STEP 2 — VALIDATING DATA")
    print("=" * 70)

    if TEXT_COLUMN not in df.columns:

        raise ValueError(
            f"\nTEXT_COLUMN = '{TEXT_COLUMN}' was not found.\n\n"
            f"Available columns are:\n{list(df.columns)}\n\n"
            "Change TEXT_COLUMN in the configuration section."
        )

    if GROUP_COLUMN is not None:

        if GROUP_COLUMN not in df.columns:

            raise ValueError(
                f"\nGROUP_COLUMN = '{GROUP_COLUMN}' was not found."
            )

    print(
        f"Text column found: {TEXT_COLUMN}"
    )

    if GROUP_COLUMN is not None:

        print(
            f"Group column found: {GROUP_COLUMN}"
        )

    return df


# =============================================================================
# 5. BASIC DATA CLEANING
# =============================================================================

def basic_data_cleaning(df):
    """
    Remove technically unusable documents.

    This step:

        1. converts missing text to empty strings
        2. converts text to string
        3. removes empty documents
        4. removes exact duplicate documents

    IMPORTANT:
    Duplicate removal should be theoretically justified.

    For example, if duplicate comments represent genuine repeated
    observations, automatically removing them may not be appropriate.

    Always consider what one row represents in your research design.
    """

    print("\n" + "=" * 70)
    print("STEP 3 — BASIC DATA CLEANING")
    print("=" * 70)

    df = df.copy()

    # -------------------------------------------------------------------------
    # Handle missing text.
    # -------------------------------------------------------------------------

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # Remove completely empty documents.
    # -------------------------------------------------------------------------

    before_empty_removal = len(df)

    df = df[
        df[TEXT_COLUMN]
        .str
        .strip()
        .ne("")
    ].copy()

    empty_removed = (
        before_empty_removal - len(df)
    )

    # -------------------------------------------------------------------------
    # Remove exact duplicate documents.
    # -------------------------------------------------------------------------

    before_duplicate_removal = len(df)

    df = df.drop_duplicates(
        subset=[TEXT_COLUMN]
    ).copy()

    duplicates_removed = (
        before_duplicate_removal - len(df)
    )

    print(
        f"Empty documents removed: {empty_removed:,}"
    )

    print(
        f"Exact duplicates removed: {duplicates_removed:,}"
    )

    print(
        f"Documents remaining: {len(df):,}"
    )

    return df


# =============================================================================
# 6. TEXT PREPROCESSING
# =============================================================================

def clean_text(text):
    """
    Perform conservative preprocessing.

    CURRENT OPERATIONS:

        - lowercase text
        - remove URLs
        - normalize whitespace

    INTENTIONALLY NOT DONE:

        - aggressive stemming
        - aggressive lemmatization
        - automatic removal of negations
        - removal of punctuation
        - removal of emojis

    WHY?

    In Human–AI research, linguistic features may themselves be theoretically meaningful.

    For example:

        "I trust the AI"

    and

        "I don't trust the AI"

    should obviously not become equivalent through careless preprocessing.

    Therefore, preprocessing should be driven by the research question,
    not simply by a generic cleaning recipe.
    """

    # Convert to lowercase.
    text = text.lower()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def preprocess_text(df):
    """
    Apply conservative preprocessing to all documents.
    """

    print("\n" + "=" * 70)
    print("STEP 4 — TEXT PREPROCESSING")
    print("=" * 70)

    df = df.copy()

    df["text_clean"] = (
        df[TEXT_COLUMN]
        .apply(clean_text)
    )

    # Remove documents that became empty.
    df = df[
        df["text_clean"]
        .str
        .strip()
        .ne("")
    ].copy()

    print(
        f"Usable documents after preprocessing: {len(df):,}"
    )

    return df


# =============================================================================
# 7. DOCUMENT LENGTH CHECK
# =============================================================================
#
# WHY?
#
# Extremely short documents can be problematic for topic modeling.
#
# A one-word comment provides very little information about its underlying
# topic distribution.
#
# However, the appropriate threshold depends on the research design.
#
# Therefore this function REPORTS document lengths rather than automatically
# deleting short documents.
# =============================================================================

def inspect_document_lengths(df):

    print("\n" + "=" * 70)
    print("STEP 5 — INSPECTING DOCUMENT LENGTH")
    print("=" * 70)

    lengths = (
        df["text_clean"]
        .str.split()
        .str.len()
    )

    print(
        f"Minimum words: {lengths.min()}"
    )

    print(
        f"Median words: {lengths.median():.0f}"
    )

    print(
        f"Mean words: {lengths.mean():.1f}"
    )

    print(
        f"Maximum words: {lengths.max()}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe script does not automatically remove short documents."
        "\nDecide on a threshold based on your research design."
    )

    return lengths


# =============================================================================
# 8. CREATE COUNT MATRIX FOR LDA
# =============================================================================
#
# LDA requires a document-term representation based on word counts.
#
# This is different from TF-IDF.
# =============================================================================

def create_count_matrix(documents):

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    return matrix, vectorizer


# =============================================================================
# 9. CREATE TF-IDF MATRIX FOR NMF
# =============================================================================

def create_tfidf_matrix(documents):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    return matrix, vectorizer


# =============================================================================
# 10. EXTRACT TOP WORDS
# =============================================================================

def extract_top_words(
    model,
    feature_names,
    n_words=N_TOP_WORDS
):
    """
    Extract the highest-weighted terms for every topic.

    The interpretation of these weights differs across algorithms.

    LDA:
        Topic-word probabilities / weights.

    NMF:
        Component loadings.

    Therefore, the resulting word lists are useful for interpretation,
    but should not be treated as identical statistical quantities.
    """

    rows = []

    for topic_index, topic in enumerate(
        model.components_
    ):

        top_indices = (
            topic
            .argsort()
            [:-n_words - 1:-1]
        )

        words = [
            feature_names[index]
            for index in top_indices
        ]

        rows.append(
            {
                "topic": topic_index,
                "top_words": ", ".join(words)
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 11. METHOD 1 — LDA
# =============================================================================
#
# WHEN SHOULD YOU USE LDA?
#
# LDA is a strong choice when:
#
#   - you want a classic and well-established method;
#   - interpretability is important;
#   - you want each document represented as a mixture of topics;
#   - you want a method that is relatively easy to explain to reviewers;
#   - you want a transparent baseline for comparison.
#
# LDA assumes that:
#
#   documents are mixtures of topics
#   and
#   topics are distributions over words.
#
# LIMITATION:
#
# LDA is based on word co-occurrence and does not directly understand
# contextual semantic similarity.
#
# Therefore, it may struggle when two texts use very different vocabulary
# to express similar meanings.
# =============================================================================

def run_lda(documents):

    print("\n" + "=" * 70)
    print("METHOD 1 — LDA")
    print("=" * 70)

    count_matrix, vectorizer = (
        create_count_matrix(documents)
    )

    print(
        f"Document-term matrix: "
        f"{count_matrix.shape[0]:,} documents × "
        f"{count_matrix.shape[1]:,} terms"
    )

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=30
    )

    document_topic_matrix = (
        lda.fit_transform(count_matrix)
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    topic_words = extract_top_words(
        lda,
        feature_names
    )

    topic_words.to_csv(
        OUTPUT_DIR / "LDA_topic_words.csv",
        index=False
    )

    document_topics = pd.DataFrame(
        document_topic_matrix,
        columns=[
            f"topic_{i}"
            for i in range(N_TOPICS)
        ]
    )

    document_topics.to_csv(
        OUTPUT_DIR / "LDA_document_topics.csv",
        index=False
    )

    return (
        lda,
        vectorizer,
        document_topic_matrix,
        topic_words
    )


# =============================================================================
# 12. METHOD 2 — NMF
# =============================================================================
#
# WHEN SHOULD YOU USE NMF?
#
# NMF is useful when:
#
#   - you want a methodologically different alternative to LDA;
#   - you use TF-IDF representations;
#   - you want relatively sparse and interpretable topics;
#   - you want a robustness analysis.
#
# A sensible research design is:
#
#       LDA = primary / baseline model
#       NMF = alternative model / robustness check
#
# Do NOT assume that agreement between methods automatically proves that
# the topics are "true". It provides evidence of robustness, but interpretation
# remains a substantive research task.
# =============================================================================

def run_nmf(documents):

    print("\n" + "=" * 70)
    print("METHOD 2 — NMF")
    print("=" * 70)

    tfidf_matrix, vectorizer = (
        create_tfidf_matrix(documents)
    )

    print(
        f"TF-IDF matrix: "
        f"{tfidf_matrix.shape[0]:,} documents × "
        f"{tfidf_matrix.shape[1]:,} terms"
    )

    nmf = NMF(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        init="nndsvda",
        max_iter=500
    )

    document_topic_matrix = (
        nmf.fit_transform(tfidf_matrix)
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    topic_words = extract_top_words(
        nmf,
        feature_names
    )

    topic_words.to_csv(
        OUTPUT_DIR / "NMF_topic_words.csv",
        index=False
    )

    document_topics = pd.DataFrame(
        document_topic_matrix,
        columns=[
            f"topic_{i}"
            for i in range(N_TOPICS)
        ]
    )

    document_topics.to_csv(
        OUTPUT_DIR / "NMF_document_topics.csv",
        index=False
    )

    return (
        nmf,
        vectorizer,
        document_topic_matrix,
        topic_words
    )


# =============================================================================
# 13. METHOD 3 — BERTOPIC
# =============================================================================
#
# WHEN SHOULD YOU USE BERTOPIC?
#
# BERTopic is particularly useful when:
#
#   - semantic similarity matters;
#   - documents are relatively short;
#   - vocabulary varies considerably;
#   - contextual meaning is important;
#   - you are working with conversational or social-media language.
#
# BERTopic typically combines:
#
#   sentence embeddings
#          ↓
#   dimensionality reduction
#          ↓
#   clustering
#          ↓
#   topic representation
#
# This is fundamentally different from LDA/NMF.
#
# IMPORTANT:
#
# BERTopic does NOT necessarily produce exactly N_TOPICS topics.
#
# Therefore:
#
#     LDA/NMF:
#         researcher specifies number of topics.
#
#     BERTopic:
#         topic structure can initially emerge from the data.
#
# This difference is methodologically important.
#
# INSTALLATION:
#
#     pip install bertopic sentence-transformers
#
# =============================================================================

def run_bertopic(documents):

    print("\n" + "=" * 70)
    print("METHOD 3 — BERTopic")
    print("=" * 70)

    try:

        from bertopic import BERTopic

    except ImportError:

        print(
            "\nBERTopic is not installed."
            "\n\nInstall it with:"
            "\n\npip install bertopic sentence-transformers"
        )

        return None

    # -------------------------------------------------------------------------
    # BASIC BERTopic CONFIGURATION
    # -------------------------------------------------------------------------
    #
    # nr_topics="auto":
    #
    # Let BERTopic determine the topic structure.
    #
    # You can alternatively use:
    #
    #     nr_topics=N_TOPICS
    #
    # but do not force the same number simply because LDA uses N_TOPICS.
    # -------------------------------------------------------------------------

    topic_model = BERTopic(
        language="english",
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True
    )

    topics, probabilities = (
        topic_model
        .fit_transform(documents)
    )

    # -------------------------------------------------------------------------
    # SAVE TOPIC INFORMATION
    # -------------------------------------------------------------------------

    topic_information = (
        topic_model
        .get_topic_info()
    )

    topic_information.to_csv(
        OUTPUT_DIR / "BERTopic_topic_information.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # SAVE DOCUMENT-LEVEL TOPIC ASSIGNMENTS
    # -------------------------------------------------------------------------

    document_topics = pd.DataFrame(
        {
            "document_id": range(
                len(documents)
            ),
            "topic": topics
        }
    )

    document_topics.to_csv(
        OUTPUT_DIR / "BERTopic_document_topics.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # EXTRACT REPRESENTATIVE WORDS
    # -------------------------------------------------------------------------

    rows = []

    for topic_id in sorted(
        set(topics)
    ):

        # BERTopic uses topic -1 for outliers.
        if topic_id == -1:
            continue

        words = (
            topic_model
            .get_topic(topic_id)
        )

        if words:

            top_words = ", ".join(
                [
                    word
                    for word, score
                    in words[:N_TOP_WORDS]
                ]
            )

            rows.append(
                {
                    "topic": topic_id,
                    "top_words": top_words
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "BERTopic_topic_words.csv",
        index=False
    )

    return topic_model


# =============================================================================
# 14. TOPIC PREVALENCE
# =============================================================================
#
# WHY?
#
# Topic modeling does not only tell us WHAT topics exist.
#
# We may also want to know how prevalent they are.
#
# For LDA:
#
#     document-topic values can be interpreted as topic proportions.
#
# For NMF:
#
#     component values are NOT probabilities.
#
# Therefore, NMF prevalence should be described as relative component
# activation / weight rather than literal probability.
# =============================================================================

def calculate_topic_prevalence(
    document_topic_matrix,
    method_name
):

    mean_topic_weights = (
        document_topic_matrix
        .mean(axis=0)
    )

    result = pd.DataFrame(
        {
            "topic": [
                f"topic_{i}"
                for i in range(
                    len(mean_topic_weights)
                )
            ],
            "mean_weight": mean_topic_weights
        }
    )

    result = (
        result
        .sort_values(
            "mean_weight",
            ascending=False
        )
    )

    result.to_csv(
        OUTPUT_DIR /
        f"{method_name}_topic_prevalence.csv",
        index=False
    )

    return result


# =============================================================================
# 15. GROUP COMPARISON
# =============================================================================
#
# OPTIONAL.
#
# Use this section if your dataset contains theoretically meaningful groups.
#
# Example:
#
# GROUP_COLUMN = "interaction_type"
#
# Values:
#
#     scrutiny
#     acceptance
#
# Then the script can calculate average topic weights for each group.
#
# IMPORTANT:
#
# A difference in topic prevalence is descriptive evidence.
#
# If you want to make inferential claims, you need an appropriate statistical
# model and should account for the structure of the data.
# =============================================================================

def compare_groups(
    df,
    document_topic_matrix,
    method_name
):

    if GROUP_COLUMN is None:

        print(
            "\nNo GROUP_COLUMN specified."
            "\nSkipping group comparison."
        )

        return None

    if GROUP_COLUMN not in df.columns:

        raise ValueError(
            f"GROUP_COLUMN '{GROUP_COLUMN}' "
            "was not found in the dataset."
        )

    topic_df = pd.DataFrame(
        document_topic_matrix,
        columns=[
            f"topic_{i}"
            for i in range(
                document_topic_matrix.shape[1]
            )
        ]
    )

    topic_df[GROUP_COLUMN] = (
        df[GROUP_COLUMN]
        .values
    )

    group_means = (
        topic_df
        .groupby(
            GROUP_COLUMN
        )
        .mean(
            numeric_only=True
        )
    )

    group_means.to_csv(
        OUTPUT_DIR /
        f"{method_name}_group_topic_means.csv"
    )

    return group_means


# =============================================================================
# 16. COMPARE DIFFERENT NUMBERS OF LDA TOPICS
# =============================================================================
#
# WHY?
#
# One of the most important decisions in topic modeling is:
#
#     "How many topics should I use?"
#
# There is no universal answer.
#
# A good research workflow combines:
#
#     1. statistical diagnostics
#     2. topic coherence
#     3. topic diversity
#     4. model stability
#     5. human interpretability
#     6. theoretical relevance
#
# Perplexity is included below as a diagnostic.
#
# IMPORTANT:
#
# Perplexity should NOT be the sole basis for selecting your final model.
# =============================================================================

def compare_lda_topic_numbers(
    documents,
    topic_numbers=(3, 5, 7, 10, 15)
):

    print("\n" + "=" * 70)
    print("STEP 6 — COMPARING DIFFERENT NUMBERS OF LDA TOPICS")
    print("=" * 70)

    count_matrix, vectorizer = (
        create_count_matrix(documents)
    )

    results = []

    for number_of_topics in topic_numbers:

        print(
            f"Testing {number_of_topics} topics..."
        )

        model = LatentDirichletAllocation(
            n_components=number_of_topics,
            random_state=RANDOM_STATE,
            learning_method="batch",
            max_iter=30
        )

        model.fit(
            count_matrix
        )

        results.append(
            {
                "n_topics":
                    number_of_topics,

                "perplexity":
                    model.perplexity(
                        count_matrix
                    ),

                "log_likelihood":
                    model.score(
                        count_matrix
                    )
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_DIR /
        "LDA_topic_number_comparison.csv",
        index=False
    )

    return results_df


# =============================================================================
# 17. SAVE LOCAL ANALYSIS DATA
# =============================================================================
#
# IMPORTANT:
#
# This function saves a local analysis copy.
#
# It is NOT intended to be committed to GitHub.
#
# Your .gitignore should later be configured so that these files are not
# accidentally uploaded.
# =============================================================================

def save_local_analysis_copy(df):

    local_path = (
        OUTPUT_DIR /
        "analysis_copy_LOCAL_ONLY.csv"
    )

    df.to_csv(
        local_path,
        index=False
    )

    print(
        f"\nLocal analysis copy saved to:\n{local_path}"
    )


# =============================================================================
# 18. MAIN PIPELINE
# =============================================================================

def main():

    print("\n")
    print("=" * 70)
    print("HUMAN–AI TEXT ANALYSIS")
    print("TOPIC MODELING TEMPLATE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1
    # -------------------------------------------------------------------------

    df = load_data(
        DATA_PATH
    )

    # -------------------------------------------------------------------------
    # STEP 2
    # -------------------------------------------------------------------------

    df = validate_data(
        df
    )

    # -------------------------------------------------------------------------
    # STEP 3
    # -------------------------------------------------------------------------

    df = basic_data_cleaning(
        df
    )

    # -------------------------------------------------------------------------
    # STEP 4
    # -------------------------------------------------------------------------

    df = preprocess_text(
        df
    )

    # -------------------------------------------------------------------------
    # STEP 5
    # -------------------------------------------------------------------------

    inspect_document_lengths(
        df
    )

    # -------------------------------------------------------------------------
    # DOCUMENT LIST
    # -------------------------------------------------------------------------

    documents = (
        df["text_clean"]
        .tolist()
    )

    if len(documents) < 20:

        raise ValueError(
            "\nThe corpus contains fewer than 20 usable documents.\n"
            "Topic modeling is unlikely to provide a meaningful structure "
            "with such a small corpus."
        )

    # -------------------------------------------------------------------------
    # METHOD 1 — LDA
    # -------------------------------------------------------------------------

    (
        lda_model,
        lda_vectorizer,
        lda_document_topics,
        lda_topic_words
    ) = run_lda(
        documents
    )

    calculate_topic_prevalence(
        lda_document_topics,
        "LDA"
    )

    compare_groups(
        df,
        lda_document_topics,
        "LDA"
    )

    # -------------------------------------------------------------------------
    # METHOD 2 — NMF
    # -------------------------------------------------------------------------

    (
        nmf_model,
        nmf_vectorizer,
        nmf_document_topics,
        nmf_topic_words
    ) = run_nmf(
        documents
    )

    calculate_topic_prevalence(
        nmf_document_topics,
        "NMF"
    )

    compare_groups(
        df,
        nmf_document_topics,
        "NMF"
    )

    # -------------------------------------------------------------------------
    # METHOD 3 — BERTopic
    # -------------------------------------------------------------------------

    bertopic_model = run_bertopic(
        documents
    )

    # -------------------------------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------------------------------

    topic_number_comparison = (
        compare_lda_topic_numbers(
            documents,
            topic_numbers=(
                3,
                5,
                7,
                10,
                15
            )
        )
    )

    # -------------------------------------------------------------------------
    # SAVE LOCAL DATA
    # -------------------------------------------------------------------------

    save_local_analysis_copy(
        df
    )

    # -------------------------------------------------------------------------
    # FINISH
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nResults have been saved locally to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not upload restricted research data or "
        "data-derived files to GitHub."
    )

    print(
        "\nRemember:"
    )

    print(
        "Topic models identify statistical structures in text."
    )

    print(
        "Researchers must interpret and validate those structures "
        "substantively."
    )


# =============================================================================
# 19. RUN THE PIPELINE
# =============================================================================

if __name__ == "__main__":

    main()
