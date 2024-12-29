import os
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import crf  # TF 1.x için CRF
from utils import createVocabulary, loadVocabulary, DataProcessor
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  
import matplotlib.pyplot as plt  



# Make sure you're on TensorFlow 1.9 for this code.
# pip install tensorflow==1.9.0

from tensorflow.contrib import crf  # CRF is in tf.contrib for TF 1.x
from utils import createVocabulary
from utils import loadVocabulary
# We don't need computeF1Score or slot-based metrics, but we keep them if the code references them.
# from utils import computeF1Score  
from utils import DataProcessor

parser = argparse.ArgumentParser(allow_abbrev=False)

# Network
parser.add_argument("--num_units", type=int, default=64, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""'full' uses slot+intent; 'intent_only' removes slot attention.""")

# Training
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false', dest='early_stop',
                    help="Disable early stopping based on intent accuracy.")
parser.add_argument("--patience", type=int, default=5, help="Patience before stopping.")

# Multi-task weighting
parser.add_argument("--slot_loss_weight", type=float, default=0.0, help="Alpha: how much slot loss contributes to total loss.")

# Label smoothing
parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for intent loss (0 = no smoothing).")

# Model/Vocab
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset name (e.g., 'atis', 'snips', or custom). Cannot be None.")
parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")

# Data
parser.add_argument("--train_data_path", type=str, default='train', help="Training data folder name.")
parser.add_argument("--test_data_path", type=str, default='test', help="Testing data folder name.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Validation data folder name.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

arg = parser.parse_args()

# Print arguments
for k, v in sorted(vars(arg).items()):
    print(k, '=', v)
print()

# Model type logic
if arg.model_type == 'full':
    add_final_state_to_intent = True
    remove_slot_attn = False
elif arg.model_type == 'intent_only':
    add_final_state_to_intent = True
    remove_slot_attn = True
else:
    print('Unknown model type:', arg.model_type)
    sys.exit(1)

# Check dataset
if arg.dataset is None:
    print('Dataset cannot be None')
    sys.exit(1)
else:
    print('Use dataset:', arg.dataset)

# Build paths
full_train_path = os.path.join('./data', arg.dataset, arg.train_data_path)
full_test_path  = os.path.join('./data', arg.dataset, arg.test_data_path)
full_valid_path = os.path.join('./data', arg.dataset, arg.valid_data_path)

# Create/load vocab
createVocabulary(os.path.join(full_train_path, arg.input_file),
                 os.path.join(arg.vocab_path, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file),
                 os.path.join(arg.vocab_path, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file),
                 os.path.join(arg.vocab_path, 'intent_vocab'))

in_vocab     = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab   = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

#################################################################
# Create Model: BiLSTM + CRF for Slots + Slot->Intent Gating
#################################################################
def createModel(
    input_data,
    input_size,
    sequence_length,
    slot_size,
    intent_size,
    layer_size=128,
    isTraining=True
):
    """
    Main model:
      - BiLSTM
      - Optional Slot Attention -> CRF
      - Slot->Intent gating
      - Intent final projection
    """
    # 1. LSTM cells
    cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    if isTraining:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5, output_keep_prob=0.5)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5, output_keep_prob=0.5)

    # 2. Embedding
    embedding = tf.get_variable('embedding', [input_size, layer_size])
    inputs = tf.nn.embedding_lookup(embedding, input_data)  # [B, T, layer_size]

    # 3. BiLSTM
    state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs, sequence_length=sequence_length, dtype=tf.float32
    )
    final_state_h = tf.concat([final_state[0][1], final_state[1][1]], axis=1)  # [B, 2*layer_size]
    state_outputs = tf.concat([state_outputs[0], state_outputs[1]], axis=2)    # [B, T, 2*layer_size]

    hidden_size = 2 * layer_size

    #################################################
    # 4. Attention (Slot + Intent)
    #################################################
    slot_inputs = state_outputs  # [B, T, 2H]

    # 4.1. Slot Attention (only if remove_slot_attn==False)
    if not remove_slot_attn:
        with tf.variable_scope('slot_attn'):
            hidden_expanded = tf.expand_dims(slot_inputs, 2)  # [B, T, 1, 2H]
            k = tf.get_variable("AttnW", [1, 1, hidden_size, hidden_size])
            hidden_features = tf.nn.conv2d(hidden_expanded, k, [1, 1, 1, 1], "SAME")
            hidden_features = tf.reshape(hidden_features, tf.shape(slot_inputs))  # [B, T, 2H]
            hidden_features = tf.expand_dims(hidden_features, 1)                  # [B, 1, T, 2H]
            v = tf.get_variable("AttnV", [hidden_size])

            # Dense transform
            slot_2d = tf.reshape(slot_inputs, [-1, hidden_size])  # [B*T, 2H]
            y = tf.layers.dense(slot_2d, hidden_size, use_bias=True, name="slot_attention_dense")
            y = tf.reshape(y, tf.shape(slot_inputs))              # [B, T, 2H]
            y = tf.expand_dims(y, 2)                              # [B, T, 1, 2H]

            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])  # [B, 1, T]
            a = tf.nn.softmax(s)                                      # [B, 1, T]
            a = tf.expand_dims(a, -1)                                 # [B, 1, T, 1]

            hidden_expanded2 = tf.expand_dims(slot_inputs, 1)         # [B, 1, T, 2H]
            slot_d = tf.reduce_sum(a * hidden_expanded2, [2])         # [B, 1, 2H]
        slot_inputs_attended = slot_d
    else:
        slot_inputs_attended = None

    # 4.2. Intent Attention
    with tf.variable_scope('intent_attn'):
        hidden_expanded_i = tf.expand_dims(state_outputs, 2)  # [B, T, 1, 2H]
        k_i = tf.get_variable("AttnW", [1, 1, hidden_size, hidden_size])
        hidden_features_i = tf.nn.conv2d(hidden_expanded_i, k_i, [1, 1, 1, 1], "SAME")
        v_i = tf.get_variable("AttnV", [hidden_size])

        # Dense transform
        y_i = tf.layers.dense(final_state_h, hidden_size, use_bias=True, name="intent_attention_dense")
        y_i = tf.reshape(y_i, [-1, 1, 1, hidden_size])  # [B, 1, 1, 2H]

        s_i = tf.reduce_sum(v_i * tf.tanh(hidden_features_i + y_i), [2, 3])  # [B, T]
        a_i = tf.nn.softmax(s_i)  # [B, T]
        a_i = tf.expand_dims(a_i, -1)
        a_i = tf.expand_dims(a_i, -1)  # [B, T, 1, 1]

        d_i = tf.reduce_sum(a_i * hidden_expanded_i, [1, 2])  # [B, 2H]

        if add_final_state_to_intent:
            intent_input = tf.concat([d_i, final_state_h], axis=1)  # [B, 4H]
        else:
            intent_input = d_i  # [B, 2H]

    # 4.3. slot_gated (Intent -> Slot)
    with tf.variable_scope('slot_gated'):
        if not remove_slot_attn:
            attn_size = hidden_size
            gate_input = tf.layers.dense(intent_input, attn_size, use_bias=True, name="intent_gate_dense")
            gate_input = tf.reshape(gate_input, [-1, 1, attn_size])  # [B, 1, 2H]
            v1 = tf.get_variable("gateV", [attn_size])

            # shape: slot_inputs_attended = [B, 1, 2H]
            slot_gate = v1 * tf.tanh(slot_inputs_attended + gate_input)  # [B, 1, 2H]
            slot_gate = tf.reduce_sum(slot_gate, [2])  # [B, 1]
            slot_gate = tf.expand_dims(slot_gate, -1)  # [B, 1, 1]
            slot_gate = slot_inputs_attended * slot_gate  # [B, 1, 2H]
            slot_gate = tf.reshape(slot_gate, [-1, attn_size])  # [B, 2H]

            slot_inputs_2 = tf.reshape(slot_inputs, [-1, attn_size])  # [B*T, 2H]
            slot_output_repr = tf.concat([slot_gate, slot_inputs_2], axis=1)  # [B*T, 4H]
        else:
            # minimal gating if remove_slot_attn = True
            slot_output_repr = tf.reshape(slot_inputs, [-1, hidden_size])

    #################################################
    # 5. Slot->Intent Gating
    #################################################
    with tf.variable_scope('slot_proj'):
        slot_logits = tf.layers.dense(slot_output_repr, slot_size, use_bias=True, name="slot_logits_dense")
    slot_logits_3d = tf.reshape(slot_logits, [tf.shape(input_data)[0], -1, slot_size])  # [B, T, slot_size]

    # Convert to probabilities, project to slot_emb, and summarize
    slot_probs_3d = tf.nn.softmax(slot_logits_3d)  # [B, T, slot_size]
    with tf.variable_scope('slot_summary_projection'):
        slot_emb_3d = tf.layers.dense(slot_probs_3d, layer_size, activation=tf.nn.relu, name="slot_summary_dense")
    slot_emb = tf.reduce_mean(slot_emb_3d, axis=1)  # [B, layer_size]

    # Combine with intent_input
    intent_input_with_slot = tf.concat([intent_input, slot_emb], axis=1)  # [B, 4H + layer_size]

    # Final intent projection
    with tf.variable_scope('intent_proj'):
        intent_logits = tf.layers.dense(intent_input_with_slot, intent_size, use_bias=True, name="intent_logits_dense")

    return slot_logits_3d, intent_logits


#################################################################
# Build Graph
#################################################################
input_data      = tf.placeholder(tf.int32,   [None, None], name='inputs')
sequence_length = tf.placeholder(tf.int32,   [None],       name="sequence_length")
global_step     = tf.Variable(0, trainable=False, name='global_step')
slots           = tf.placeholder(tf.int32,   [None, None], name='slots')
intent          = tf.placeholder(tf.int32,   [None],       name='intent')

# --- Create training model ---
with tf.variable_scope('model'):
    slot_logits_3d, intent_logits = createModel(
        input_data,
        len(in_vocab['vocab']),
        sequence_length,
        len(slot_vocab['vocab']),
        len(intent_vocab['vocab']),
        layer_size=arg.layer_size,
        isTraining=True
    )

# -------------------------
# 1. CRF Loss for Slots
# -------------------------
with tf.variable_scope('slot_crf'):
    log_likelihood, crf_trans_params = crf.crf_log_likelihood(
        slot_logits_3d,
        tag_indices=slots,
        sequence_lengths=sequence_length
    )
    slot_loss = tf.reduce_mean(-log_likelihood)

# -------------------------
# 2. Intent Loss (with label smoothing)
# -------------------------
def label_smoothing_loss(logits, labels, smoothing, vocab_size):
    """
    Manual label smoothing for a multi-class classification.
    labels are a 1D tensor of shape [B].
    """
    # Convert to one-hot
    labels_onehot = tf.one_hot(labels, depth=vocab_size)  # shape [B, vocab_size]
    # Smooth the one-hot distribution
    smooth_pos = 1.0 - smoothing
    smooth_neg = smoothing / float(vocab_size - 1)
    labels_smoothed = labels_onehot * smooth_pos + (1 - labels_onehot) * smooth_neg
    # Compute cross-entropy
    xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_smoothed)
    return tf.reduce_mean(xent)

with tf.variable_scope('intent_loss'):
    # Weighted label smoothing
    intent_loss = label_smoothing_loss(
        logits=intent_logits,
        labels=intent,
        smoothing=arg.label_smoothing,
        vocab_size=len(intent_vocab['vocab'])
    )

# -------------------------
# 3. Multi-Task Combined Loss
# -------------------------
alpha = arg.slot_loss_weight  # how much slot loss matters
loss = alpha * slot_loss + (1.0 - alpha) * intent_loss

# -------------------------
# 4. Optimizer & Learning Rate Decay
# -------------------------
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True
)
opt = tf.train.AdamOptimizer(learning_rate)

# Gradient clipping
params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

training_outputs = [global_step, slot_loss, intent_loss, loss, train_op]

# -------------------------
# 5. Inference Model (reuse weights)
# -------------------------
with tf.variable_scope('model', reuse=True):
    slot_logits_3d_inf, intent_logits_inf = createModel(
        input_data,
        len(in_vocab['vocab']),
        sequence_length,
        len(slot_vocab['vocab']),
        len(intent_vocab['vocab']),
        layer_size=arg.layer_size,
        isTraining=False
    )

with tf.variable_scope('slot_crf', reuse=True):
    # CRF decode
    viterbi_seq, viterbi_scores = crf.crf_decode(
        slot_logits_3d_inf,
        crf_trans_params,
        sequence_length
    )

# Note: We do not actually evaluate slot predictions in this script,
# but we keep them if they help the model learn better.
inference_slot_output   = viterbi_seq  # [B, T]
inference_intent_output = tf.nn.softmax(intent_logits_inf, name='intent_output')

# We only care about INTENT accuracy in evaluation
inference_outputs = [inference_intent_output]
inference_inputs  = [input_data, sequence_length]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
saver = tf.train.Saver()

#################################################################
# Training
#################################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    logging.info('Training Start')

    epochs = 0
    sum_slot_loss = 0.0
    sum_intent_loss = 0.0
    num_batches = 0
    step = 0
    no_improve = 0

    best_valid_intent_acc = 0.0
    final_test_acc = 0.0  # <-- We'll store the last test accuracy here.
    data_processor = None

    while True:
        # ----- Get a batch of data -----
        if data_processor is None:
            data_processor = DataProcessor(
                os.path.join(full_train_path, arg.input_file),
                os.path.join(full_train_path, arg.slot_file),
                os.path.join(full_train_path, arg.intent_file),
                in_vocab, slot_vocab, intent_vocab
            )
        in_data, slot_data_, slot_weight_data, length_data, intent_data_, _, _, _ = data_processor.get_batch(arg.batch_size)

        feed_dict = {
            input_data: in_data,
            slots: slot_data_,
            sequence_length: length_data,
            intent: intent_data_
        }

        ret = sess.run(training_outputs, feed_dict)
        step_val, s_loss_val, i_loss_val, total_loss_val, _ = ret

        sum_slot_loss += s_loss_val
        sum_intent_loss += i_loss_val
        num_batches += 1

        # End of epoch?
        if data_processor.end == 1:
            epochs += 1
            avg_slot_loss = sum_slot_loss / num_batches
            avg_intent_loss = sum_intent_loss / num_batches
            logging.info(
                'Epoch {} | step={} | slot_loss={:.4f} | intent_loss={:.4f} | alpha={:.2f}'.format(
                    epochs, step_val, avg_slot_loss, avg_intent_loss, alpha
                )
            )
            sum_slot_loss, sum_intent_loss, num_batches = 0.0, 0.0, 0

            data_processor.close()
            data_processor = None

            # -------------- Validation on INTENT only --------------
            def run_intent_evaluation(in_path, slot_path, intent_path):
                dp_val = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab)
                pred_intents = []
                correct_intents = []
                while True:
                    in_data_v, slot_data_v, slot_weight_v, length_v, intent_data_v, _, _, _ = dp_val.get_batch(arg.batch_size)
                    if in_data_v.shape[0] == 0:
                        break
                    feed_dict_v = {input_data: in_data_v, sequence_length: length_v}
                    [intent_probs_v] = sess.run(inference_outputs, feed_dict_v)

                    for row in intent_probs_v:
                        pred_intents.append(np.argmax(row))
                    correct_intents.extend(intent_data_v)

                    if dp_val.end == 1:
                        break
                dp_val.close()

                pred_intents_arr = np.array(pred_intents)
                correct_intents_arr = np.array(correct_intents)
                acc_intent = (pred_intents_arr == correct_intents_arr).mean() * 100.0
                return acc_intent

            # Evaluate on validation
            val_intent_acc = run_intent_evaluation(
                os.path.join(full_valid_path, arg.input_file),
                os.path.join(full_valid_path, arg.slot_file),
                os.path.join(full_valid_path, arg.intent_file)
            )
            logging.info("Validation intent_acc={:.2f}".format(val_intent_acc))

            # Evaluate on test (just for reference)
            test_intent_acc = run_intent_evaluation(
                os.path.join(full_test_path, arg.input_file),
                os.path.join(full_test_path, arg.slot_file),
                os.path.join(full_test_path, arg.intent_file)
            )
            logging.info("Test intent_acc={:.2f}".format(test_intent_acc))

            # Keep track of the last test accuracy
            final_test_acc = test_intent_acc

            # Save model
            saver.save(sess, os.path.join(arg.model_path, 'model.ckpt'), global_step=epochs)

            # Early stopping based on intent accuracy
            if val_intent_acc <= best_valid_intent_acc:
                no_improve += 1
            else:
                best_valid_intent_acc = val_intent_acc
                no_improve = 0

            if epochs >= arg.max_epochs:
                logging.info("Reached max epochs. Stop.")
                break

            if arg.early_stop and no_improve > arg.patience:
                logging.info("No improvement for {} checks, early stop.".format(arg.patience))
                break

logging.info("Training Completed.")

# -----------------------------------------
# Print the final test accuracy at the end
# -----------------------------------------

print("Final Test Intent Accuracy: {:.2f}".format(final_test_acc))



output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)  


report = classification_report(correct_intents, pred_intents, target_names=intent_vocab['vocab'])

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write("Classification Report:\n")
    f.write(report)


conf_matrix = confusion_matrix(correct_intents, pred_intents)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=intent_vocab['vocab'])
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")

plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()  
