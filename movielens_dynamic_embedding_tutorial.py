# %%
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
import tensorflow_datasets as tfds
import functools
from typing import Dict
import dataclasses
print(tf.__version__)

# %%
"""
## Download datasets
"""

# %%
# https://www.tensorflow.org/datasets/catalog/movielens
# Interactions dataset
raw_ratings_dataset = tfds.load("movielens/1m-ratings", split="train")
# Candidates dataset
raw_movies_dataset = tfds.load("movielens/1m-movies", split="train")

# %%
df = tfds.as_dataframe(raw_ratings_dataset.take(100))
df.head()

# %%
for item in raw_ratings_dataset.take(1):
    print(item)

# %%
"""
## Processing datasets
"""

# %%
max_token_length = 10
pad_token = "[PAD]"
punctuation_regex = "[\!\"#\$%&\(\)\*\+,-\.\/\:;\<\=\>\?@\[\]\\\^_`\{\|\}~\\t\\n]"

def process_text(x: tf.Tensor, max_token_length: int, punctuation_regex: str) -> tf.Tensor:
    
    return tf.strings.split(
        tf.strings.regex_replace(
            tf.strings.lower(x["movie_title"]), punctuation_regex, ""
        )
    )[:max_token_length]


def process_ratings_dataset(ratings_dataset: tf.data.Dataset) -> tf.data.Dataset:
    
    partial_process_text = functools.partial(
        process_text, max_token_length=max_token_length, punctuation_regex=punctuation_regex
    )

    preprocessed_movie_title_dataset = ratings_dataset.map(
        lambda x: partial_process_text(x)
    )

    processed_dataset = tf.data.Dataset.zip(
        (ratings_dataset, preprocessed_movie_title_dataset)
    ).map(
        lambda x,y: {"user_id": x["user_id"]} | {"movie_title": y}
    )
    
    return processed_dataset


def process_movies_dataset(movies_dataset: tf.data.Dataset) -> tf.data.Dataset:
    
    partial_process_text = functools.partial(
        process_text, max_token_length=max_token_length, punctuation_regex=punctuation_regex
    )
    
    processed_dataset = raw_movies_dataset.map(
        lambda x: partial_process_text(x)
    )
    
    return processed_dataset

processed_ratings_dataset = process_ratings_dataset(raw_ratings_dataset)
for item in processed_ratings_dataset.take(3):
    print(item)

# %%
batch_size=4096
seed=2023
train_size = int(len(processed_ratings_dataset) * 0.9)
validation_size = len(processed_ratings_dataset) - train_size
print(f"Train size: {train_size}")
print(f"Validation size: {validation_size}")

# Things we're not considering
# out of sample validation
# out of time validation

@dataclasses.dataclass(frozen=True)
class TrainingDatasets:
    train_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset

@dataclasses.dataclass(frozen=True)
class RetrievalDatasets:
    training_datasets: TrainingDatasets
    candidate_dataset: tf.data.Dataset

def pad_and_batch_ratings_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    
    return dataset.padded_batch(
        batch_size, 
        padded_shapes={
            "user_id": tf.TensorShape([]),
            "movie_title": tf.TensorShape([max_token_length,])
        }, padding_values={
            "user_id": pad_token, 
            "movie_title": pad_token
        }
    )
    
        
def split_train_validation_datasets(ratings_dataset: tf.data.Dataset) -> TrainingDatasets:
    
    shuffled_dataset = ratings_dataset.shuffle(buffer_size=5*batch_size, seed=seed)
    train_ds = shuffled_dataset.skip(validation_size).shuffle(buffer_size=10*batch_size).apply(pad_and_batch_ratings_dataset)
    validation_ds = shuffled_dataset.take(validation_size).apply(pad_and_batch_ratings_dataset)
    
    return TrainingDatasets(train_ds=train_ds, validation_ds=validation_ds)

def pad_and_batch_candidate_dataset(movies_dataset: tf.data.Dataset) -> tf.data.Dataset:
    return movies_dataset.padded_batch(
        batch_size, 
        padded_shapes=tf.TensorShape([max_token_length,]), 
        padding_values=pad_token
    )

def create_datasets() -> RetrievalDatasets:
    
    raw_ratings_dataset = tfds.load("movielens/1m-ratings", split="train")
    raw_movies_dataset = tfds.load("movielens/1m-movies", split="train")
    
    processed_ratings_dataset = process_ratings_dataset(raw_ratings_dataset)
    processed_movies_dataset = process_movies_dataset(raw_movies_dataset)

    training_datasets = split_train_validation_datasets(processed_ratings_dataset)
    candidate_dataset = pad_and_batch_candidate_dataset(processed_movies_dataset)
    
    return RetrievalDatasets(training_datasets=training_datasets, candidate_dataset=candidate_dataset)

datasets = create_datasets()
print(f"Train dataset size (after batching): {len(datasets.training_datasets.train_ds)}")
print(f"Validation dataset size (after batching): {len(datasets.training_datasets.validation_ds)}")

# %%
"""
## Defining user and item towers
"""

# %%
def get_user_id_lookup_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Layer:
    user_lookup_layer = tf.keras.layers.StringLookup(mask_token=None)
    user_lookup_layer.adapt(dataset.map(lambda x: x["user_id"]))
    return user_lookup_layer

def build_user_model(user_id_lookup_layer: tf.keras.layers.StringLookup):
    vocab_size = user_id_lookup_layer.vocabulary_size()
    return tf.keras.Sequential([
        # Fix from https://github.com/keras-team/keras/issues/16101
        tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
        user_id_lookup_layer, 
        tf.keras.layers.Embedding(vocab_size, 64), 
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        
    ], name='user_model')


# %%
def get_movie_title_lookup_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Layer:
    movie_title_lookup_layer = tf.keras.layers.StringLookup(mask_token=pad_token)#, input_shape=[max_token_length,])
    movie_title_lookup_layer.adapt(dataset.map(lambda x: x["movie_title"]))
    return movie_title_lookup_layer

def build_item_model(movie_title_lookup_layer: tf.keras.layers.StringLookup):
    vocab_size = movie_title_lookup_layer.vocabulary_size()
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_token_length), dtype=tf.string),
        movie_title_lookup_layer, 
        tf.keras.layers.Embedding(vocab_size, 64), 
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

# %%
"""
## Testing the lookup layers
"""

# %%
# TODO: example models
user_id_lookup_layer = get_user_id_lookup_layer(datasets.training_datasets.train_ds)
print(f"Vocabulary size (user_ids): {len(user_id_lookup_layer.get_vocabulary())}")

# %%
"""
## Defining the two tower model (without dynamic embeddings)
"""

# %%
class TwoTowerModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

    def __init__(self,user_model: tf.keras.Model,item_model: tf.keras.Model,task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.item_model = item_model
        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.item_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)


# %%
"""
## Creating the model
"""

# %%
def create_two_tower_model(dataset: tf.data.Dataset, candidate_dataset: tf.data.Dataset) -> tf.keras.Model:
    user_id_lookup_layer = get_user_id_lookup_layer(dataset)
    movie_title_lookup_layer = get_movie_title_lookup_layer(dataset)
    user_model = build_user_model(user_id_lookup_layer)
    item_model = build_item_model(movie_title_lookup_layer)
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidate_dataset.map(item_model)
        ),
    )

    model = TwoTowerModel(user_model, item_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    
    return model

datasets = create_datasets()
model = create_two_tower_model(datasets.training_datasets.train_ds, datasets.candidate_dataset)

# %%
"""
## Training the model
"""


print("Training the model")

# %%
# Train for 3 epochs.

# history = model.fit(datasets.training_datasets.train_ds, epochs=3, validation_data=datasets.training_datasets.validation_ds)

# %%
# factorized_top_k/top_100_categorical_accuracy
# val_factorized_top_k/top_100_categorical_accuracy
# history_standard = history.history

# %%
"""
## Using `BasicEmbedding` layer

Defining user tower and iterm tower with dynamic embeddings
"""

# %%
tf.compat.v1.reset_default_graph()

class UniqueEmbedding(de.keras.layers.BasicEmbedding):
  """
  The UniqueEmbedding layer has the same input and output with BasicEmbedding,
  but it will do a unique operator before embedding_lookup.
  ```
  """

  def call(self, ids):
    with tf.name_scope(self.name + "/EmbeddingLookupUnique"):
      ids = tf.convert_to_tensor(ids)
      shape = tf.shape(ids)
      ids_flat = tf.reshape(ids, tf.reduce_prod(shape, keepdims=True))
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = de.shadow_ops.embedding_lookup(
          self.shadow, unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
      embeddings_shape = tf.concat([tf.shape(ids), [self.embedding_size]], 0)
      embeddings = tf.reshape(embeddings_flat, embeddings_shape)
      return embeddings

def build_de_user_model(user_id_lookup_layer: tf.keras.layers.StringLookup) -> tf.keras.layers.Layer:
    vocab_size = user_id_lookup_layer.vocabulary_size()
    return tf.keras.Sequential([
        # Fix from https://github.com/keras-team/keras/issues/16101
        tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
        user_id_lookup_layer, 
        UniqueEmbedding(
            embedding_size=64,
            initializer=tf.random_uniform_initializer(),
            init_capacity=vocab_size*2.0,
            restrict_policy=de.FrequencyRestrictPolicy,
            name="UserDynamicEmbeddingLayer"
        ), 
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        
    ], name='user_model')

def build_de_item_model(movie_title_lookup_layer: tf.keras.layers.StringLookup) -> tf.keras.layers.Layer:
    vocab_size = movie_title_lookup_layer.vocabulary_size()
    print("vocab_size = ", vocab_size)
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_token_length), dtype=tf.string),
        movie_title_lookup_layer, 
        UniqueEmbedding(
            embedding_size=64,
            initializer=tf.random_uniform_initializer(),
            init_capacity=vocab_size*2.0,
            restrict_policy=de.FrequencyRestrictPolicy,
            name="ItemDynamicEmbeddingLayer"
        ), 
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

# %%
"""
## Defining the callback to control and log dynamic embeddings
"""

# %%
class DynamicEmbeddingCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, model, steps_per_logging, steps_per_restrict):
        self.model = model
        self.steps_per_logging = steps_per_logging
        self.steps_per_restrict = steps_per_restrict
    
    def on_train_begin(self, logs=None):
        self.model.dynamic_embedding_history = {}
        
    def on_train_batch_end(self, batch, logs=None):
        
        if batch and batch % self.steps_per_logging == 0:
            if logs:
                
                embedding_size_dict = {
                    k:self.model.embedding_layers[k].params.size().numpy() 
                    for k in self.model.embedding_layers.keys()
                }
                
                for k, v in embedding_size_dict.items():
                    self.model.dynamic_embedding_history.setdefault(f"embedding_size_{k}", []).append(v)
                self.model.dynamic_embedding_history.setdefault(f"step", []).append(batch)
                
        # if batch and batch % self.steps_per_restrict == 0:
        #
        #     [
        #         self.model.embedding_layers[k].params.restrict(
        #             int(self.model.lookup_vocab_sizes[k]*0.95),
        #             trigger=int(self.model.lookup_vocab_sizes[k]*1.0)
        #         ) for k in self.model.embedding_layers.keys()
        #     ]

# %%
"""
## Defining two tower model with dynamic embeddings
"""

# %%
class DynamicEmbeddingTwoTowerModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

    def __init__(self,user_model: tf.keras.Model,item_model: tf.keras.Model,task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.item_model = item_model
        
        self.embedding_layers = {
            "user": user_model.layers[1], 
            "movie": item_model.layers[1]
        }
        
        if not all(["embedding" in v.name.lower() for k,v in self.embedding_layers.items()]):
            raise TypeError(
                f"""All layers in embedding_layers must be embedding layers.
                Got {[v.name for v in self.embedding_layers.values()]}"""
            )
            
        self.lookup_vocab_sizes = {
            "user": user_model.layers[0].vocabulary_size(), 
            "movie": item_model.layers[0].vocabulary_size()
        }
        self.dynamic_embedding_history = {}
        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.item_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)


# %%
def create_de_two_tower_model(dataset: tf.data.Dataset, candidate_dataset: tf.data.Dataset) -> tf.keras.Model:
    
    user_id_lookup_layer = get_user_id_lookup_layer(dataset)
    movie_title_lookup_layer = get_movie_title_lookup_layer(dataset)
    user_model = build_de_user_model(user_id_lookup_layer)
    item_model = build_de_item_model(movie_title_lookup_layer)
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidate_dataset.map(item_model)
        ),
    )

    model = DynamicEmbeddingTwoTowerModel(user_model, item_model, task)
    optimizer = de.DynamicEmbeddingOptimizer(tf.keras.optimizers.Adam())
    model.compile(optimizer=optimizer)
    
    return model

datasets = create_datasets()
de_model = create_de_two_tower_model(datasets.training_datasets.train_ds, datasets.candidate_dataset)

# %%
"""
## Training the model with dynamic embeddings (streaming)
"""
print("Training the model with dynamic embeddings (streaming)")
# %%
epochs = 3
# TODO: enable shuffle after testing

history_de = {}
history_de_size = {}
de_callback = DynamicEmbeddingCallback(de_model, steps_per_logging=20, steps_per_restrict=100)

for epoch in range(epochs):

    datasets = create_datasets()
    train_steps = len(datasets.training_datasets.train_ds)
    
    hist = de_model.fit(
        datasets.training_datasets.train_ds, 
        epochs=1, 
        validation_data=datasets.training_datasets.validation_ds, 
        callbacks=[de_callback]
    )
    
    for k,v in de_model.dynamic_embedding_history.items():
        if k=="step":
            v = [vv+(epoch*train_steps) for vv in v]
        history_de_size.setdefault(k, []).extend(v)
        
    for k,v in hist.history.items():
        history_de.setdefault(k, []).extend(v)

# %%
"""
## Plotting embedding sizes
"""

# %%
import matplotlib.pyplot as plt
# %matplotlib inline

steps = history_de_size["step"]
plt.plot(steps, history_de_size["embedding_size_user"], color="r", label="User embedding")
plt.axhline(de_model.lookup_vocab_sizes["user"], color="r", linestyle="--")
plt.plot(steps, history_de_size["embedding_size_movie"], color="g", label="Movie embedding")
plt.axhline(de_model.lookup_vocab_sizes["movie"], color="g", linestyle="--")
plt.ylabel("Embedding size")
plt.xlabel("Step")
plt.title("Changes in the embedding size over time")
plt.legend()
plt.show()

# %%
"""
## Plotting accuracies
"""

# %%
# factorized_top_k/top_100_categorical_accuracy
# val_factorized_top_k/top_100_categorical_accuracy
steps = [(epoch+1)*train_steps for epoch in range(epochs)]
plt.plot(steps, history_standard["factorized_top_k/top_100_categorical_accuracy"], color="r", linestyle="--", label="Train accuracy")
plt.plot(steps, history_de["factorized_top_k/top_100_categorical_accuracy"], color="g", linestyle="--", label="Train accuracy (DE)")
plt.plot(steps, history_standard["val_factorized_top_k/top_100_categorical_accuracy"], color="r", label="Validation accuracy")
plt.plot(steps, history_de["val_factorized_top_k/top_100_categorical_accuracy"], color="g", label="Validation accuracy (DE)")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Model accuracy with and wo dynamic embeddings")

# %%
"""
## Leftovers from experiments
"""

# %%
class EvaluatorCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, model, validation_data, steps_per_validation):
        self.model = model
        self.validation_data = validation_data
        self.steps_per_validation = steps_per_validation
    
    def on_train_begin(self, logs=None):
        self.model.streaming_history = {}
    
    def on_train_batch_end(self, batch, logs=None):
        
        if batch and batch % self.steps_per_validation == 0:
            
            #tf.keras.io_utils.print_msg('\n')
            print("") # 
            val_logs = self.model.evaluate(self.validation_data, return_dict=True)
            #print("\nval logs", val_logs)
            val_logs = {
                "val_" + name: val for name, val in val_logs.items()
            }
            logs.update(val_logs)
            
            self.model.streaming_history.setdefault("step", []).append(batch)
            for k, v in logs.items():
                self.model.streaming_history.setdefault(k, []).append(v)
                
            print("\n", self.model.streaming_history)


# %%


# %%


# %%


# %%
