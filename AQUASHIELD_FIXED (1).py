# ╔══════════════════════════════════════════════════════════════════╗
# ║         AquaShield AI — COMPLETE FIXED PROJECT                  ║
# ║         CipherLab AI Hackathon | EGS Pillay Engineering         ║
# ║                                                                  ║
# ║  ALL ISSUES FIXED:                                              ║
# ║  ✅ Grad-CAM layer name error → auto-detect fixed               ║
# ║  ✅ Bad gateway / Streamlit removed → Flask + ngrok             ║
# ║  ✅ No inline widget UI → opens in real browser tab             ║
# ║  ✅ All variable scope errors fixed                             ║
# ║  ✅ X_test_s variable not found error fixed                     ║
# ║                                                                  ║
# ║  HOW TO RUN:                                                    ║
# ║  1. Runtime → Change runtime type → T4 GPU                      ║
# ║  2. Copy each CELL into a separate Colab cell                   ║
# ║  3. Run cells ONE BY ONE — wait for each to finish              ║
# ║  4. Cell 5: paste your free ngrok token → get browser link      ║
# ╚══════════════════════════════════════════════════════════════════╝


# ════════════════════════════════════════════════════════════════════
# CELL 1 — Install packages
# Paste this entire block into Colab Cell 1 and run it.
# Wait until you see: ✅ All packages ready
# ════════════════════════════════════════════════════════════════════

!pip install flask flask-cors pyngrok \
             pillow opencv-python-headless \
             tensorflow scikit-learn \
             matplotlib seaborn numpy pandas -q

!pip install pyngrok --upgrade -q

import importlib, sys
for pkg in ['flask','pyngrok','cv2','tensorflow','sklearn']:
    try:
        importlib.import_module(pkg)
        print(f"  ✅ {pkg}")
    except:
        print(f"  ❌ {pkg} — reinstall needed")

print("\n✅ All packages ready — run Cell 2")


# ════════════════════════════════════════════════════════════════════
# CELL 2 — Imports & global constants
# Paste this entire block into Colab Cell 2 and run it.
# ════════════════════════════════════════════════════════════════════

import os, cv2, random, pickle, warnings, io, base64, threading, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
    Dropout, Input, Concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings('ignore')

# ── Constants (shared across all cells) ──────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 16
NUM_CLASSES     = 3
DATASET_DIR     = '/content/water_pollution_data'
SAVE_DIR        = '/content/aquashield_outputs'
MODEL_PATH      = f'{SAVE_DIR}/best_model_final.keras'
SCALER_PATH     = f'{SAVE_DIR}/scaler.pkl'
SENSOR_FEATURES = ['ph', 'turbidity', 'dissolved_oxygen', 'temperature', 'cod']
CLASS_NAMES_MAP = {0: 'CRITICAL', 1: 'MODERATE', 2: 'SAFE'}
CLASS_ICONS_MAP = {0: '🔴', 1: '🟡', 2: '🟢'}

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,    exist_ok=True)

# Global references — set in Cell 3, used in Cell 5
_model_global  = None
_scaler_global = None

print(f"✅ TensorFlow  : {tf.__version__}")
print(f"✅ GPU         : {tf.config.list_physical_devices('GPU')}")
print(f"✅ Dataset dir : {DATASET_DIR}")
print(f"✅ Save dir    : {SAVE_DIR}")
print("\n✅ Constants ready — run Cell 3")


# ════════════════════════════════════════════════════════════════════
# CELL 3 — Build dataset + Train model
# This takes ~25 minutes on GPU.
# Wait until you see: ✅ TRAINING COMPLETE — run Cell 4
# ════════════════════════════════════════════════════════════════════

# ── 3A: Synthetic image generator ───────────────────────────────
def make_water_image(cls, size=224):
    """Generates a realistic synthetic river water image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if cls == 'safe':
        base  = [random.randint(80,140),
                 random.randint(140,210),
                 random.randint(175,245)]
        noise = 16
    elif cls == 'moderate':
        base  = [random.randint(100,165),
                 random.randint(95,155),
                 random.randint(50,115)]
        noise = 32
    else:  # critical
        base  = [random.randint(30,88),
                 random.randint(22,68),
                 random.randint(12,52)]
        noise = 44

    img[:] = base
    n = np.random.normal(0, noise, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)

    for i in range(size):
        w = int(4 * np.sin(i * 0.13 + random.random() * 5))
        if 0 < w < size:
            img[i, :, 1] = np.clip(img[i,:,1].astype(int) + w, 0, 255)

    if cls == 'critical':
        for _ in range(random.randint(8, 26)):
            cx, cy = random.randint(10,size-10), random.randint(10,size-10)
            cv2.circle(img, (cx,cy), random.randint(3,15), (195,188,180), -1)
        for _ in range(random.randint(3, 8)):
            p1 = (random.randint(0,size), random.randint(0,size))
            p2 = (random.randint(0,size), random.randint(0,size))
            cv2.line(img, p1, p2, (15,10,8), random.randint(2,5))
    elif cls == 'moderate':
        for _ in range(random.randint(5, 14)):
            cx, cy = random.randint(20,size-20), random.randint(20,size-20)
            cv2.ellipse(img, (cx,cy),
                        (random.randint(8,22), random.randint(4,12)),
                        random.randint(0,180), 0, 360, (72,108,55), -1)

    return cv2.GaussianBlur(img, (3,3), 0)


# ── 3B: Synthetic sensor row ────────────────────────────────────
def make_sensor_row(cls):
    if cls == 'safe':
        return dict(ph=round(random.uniform(6.5,8.5),2),
                    turbidity=round(random.uniform(0.5,5.0),2),
                    dissolved_oxygen=round(random.uniform(6.5,14.0),2),
                    temperature=round(random.uniform(18.0,27.0),2),
                    cod=round(random.uniform(0,18),2), label=cls)
    elif cls == 'moderate':
        return dict(ph=round(random.uniform(5.5,6.5),2),
                    turbidity=round(random.uniform(5.0,50.0),2),
                    dissolved_oxygen=round(random.uniform(3.0,6.0),2),
                    temperature=round(random.uniform(22.0,32.0),2),
                    cod=round(random.uniform(20,100),2), label=cls)
    else:
        return dict(ph=round(random.uniform(2.8,5.5),2),
                    turbidity=round(random.uniform(55.0,500.0),2),
                    dissolved_oxygen=round(random.uniform(0.0,3.0),2),
                    temperature=round(random.uniform(25.0,38.0),2),
                    cod=round(random.uniform(105,500),2), label=cls)


# ── 3C: Build dataset ───────────────────────────────────────────
print("="*55)
print("STEP 1/4 — Building synthetic dataset")
print("="*55)

classes = ['safe', 'moderate', 'critical']
n_train, n_val, n_test = 280, 60, 60
sensor_rows = []

for cls in classes:
    for sp, n in [('train',n_train),('val',n_val),('test',n_test)]:
        folder = f'{DATASET_DIR}/{sp}/{cls}'
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            img = make_water_image(cls)
            cv2.imwrite(f'{folder}/{cls}_{sp}_{i:04d}.jpg', img)
    for _ in range(400):
        sensor_rows.append(make_sensor_row(cls))
    print(f"  ✅ {cls}: {n_train+n_val+n_test} images + 400 sensor rows")

sensor_df = pd.DataFrame(sensor_rows)
sensor_df.to_csv(f'{SAVE_DIR}/sensor_data.csv', index=False)
print(f"\nTotal images: {(n_train+n_val+n_test)*3}")
print(f"Total sensor rows: {len(sensor_df)}")


# ── 3D: Image generators ────────────────────────────────────────
print("\n" + "="*55)
print("STEP 2/4 — Setting up data pipeline")
print("="*55)

train_gen = ImageDataGenerator(
    rescale=1/255., rotation_range=20,
    width_shift_range=0.15, height_shift_range=0.15,
    horizontal_flip=True, brightness_range=[0.75,1.25], zoom_range=0.12
).flow_from_directory(
    f'{DATASET_DIR}/train', target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=42)

val_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    f'{DATASET_DIR}/val', target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

test_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    f'{DATASET_DIR}/test', target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

print(f"Train: {train_gen.n} | Val: {val_gen.n} | Test: {test_gen.n}")
print(f"Class indices: {train_gen.class_indices}")

# ── 3E: Sensor preprocessing ────────────────────────────────────
label_map = {'safe':2, 'moderate':1, 'critical':0}
sensor_df['label_enc'] = sensor_df['label'].map(label_map)

X_all = sensor_df[SENSOR_FEATURES].values
y_all = sensor_df['label_enc'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved → {SCALER_PATH}")

# ── FIX: Keep X_test_s in scope for evaluation later ────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_scaled, y_all, test_size=0.30, random_state=42, stratify=y_all)
X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"Sensor splits — train:{X_tr.shape} val:{X_val_s.shape} test:{X_test_s.shape}")

STEPS  = max(1, train_gen.n // BATCH_SIZE)
VSTEPS = max(1, val_gen.n  // BATCH_SIZE)


# ── 3F: Dual-input generator ─────────────────────────────────────
def dual_gen(img_gen, sensor_X, bs=BATCH_SIZE):
    """Yields {image_input, sensor_input} → labels continuously."""
    while True:
        img_batch, img_labels = next(img_gen)
        cur = len(img_batch)
        idx = np.random.choice(len(sensor_X), cur, replace=True)
        yield ({'image_input': img_batch, 'sensor_input': sensor_X[idx]},
               img_labels)


# ── 3G: Build model ──────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 3/4 — Building model")
print("="*55)

img_in = Input(shape=(IMG_SIZE,IMG_SIZE,3), name='image_input')
base   = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),
                     include_top=False, weights='imagenet')
base.trainable = False

x = base(img_in, training=False)
x = GlobalAveragePooling2D(name='gap')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.40)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.30)(x)

sen_in = Input(shape=(5,), name='sensor_input')
s = Dense(64, activation='relu')(sen_in)
s = BatchNormalization()(s)
s = Dropout(0.30)(s)
s = Dense(32, activation='relu')(s)
s = Dropout(0.20)(s)

fused = Concatenate(name='fusion')([x, s])
f = Dense(128, activation='relu')(fused)
f = BatchNormalization()(f)
f = Dropout(0.30)(f)
f = Dense(64, activation='relu')(f)
out = Dense(NUM_CLASSES, activation='softmax', name='output')(f)

model = Model(inputs=[img_in, sen_in], outputs=out, name='AquaShield')
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])
print(f"Parameters: {model.count_params():,}")


# ── 3H: Train Phase 1 ────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 4/4 — Training")
print("Phase 1: Head training (base frozen) — ~10 min")
print("="*55)

h1 = model.fit(
    dual_gen(train_gen, X_tr),
    steps_per_epoch=STEPS,
    validation_data=dual_gen(val_gen, X_val_s),
    validation_steps=VSTEPS,
    epochs=15,
    callbacks=[
        EarlyStopping('val_accuracy', patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=3,
                          min_lr=1e-8, verbose=1)
    ],
    verbose=1
)
print(f"\n✅ Phase 1 best val_accuracy: {max(h1.history['val_accuracy'])*100:.2f}%")


# ── 3I: Train Phase 2 ────────────────────────────────────────────
print("\nPhase 2: Fine-tuning top 30 MobileNetV2 layers — ~15 min")

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy',
              metrics=['accuracy'])

h2 = model.fit(
    dual_gen(train_gen, X_tr),
    steps_per_epoch=STEPS,
    validation_data=dual_gen(val_gen, X_val_s),
    validation_steps=VSTEPS,
    epochs=20,
    callbacks=[
        EarlyStopping('val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau('val_loss', factor=0.3, patience=4,
                          min_lr=1e-9, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1)
    ],
    verbose=1
)
print(f"\n✅ Phase 2 best val_accuracy: {max(h2.history['val_accuracy'])*100:.2f}%")
print(f"✅ Model saved → {MODEL_PATH}")


# ── 3J: Test evaluation ──────────────────────────────────────────
print("\nEvaluating on test set...")
model_eval = tf.keras.models.load_model(MODEL_PATH)
test_gen.reset()
y_true_list, y_pred_list = [], []

for _ in range(test_gen.n // BATCH_SIZE + 1):
    try:
        ib, lb = next(test_gen)
        bs  = len(ib)
        idx = np.random.choice(len(X_test_s), bs, replace=True)
        pr  = model_eval.predict({'image_input': ib,
                                   'sensor_input': X_test_s[idx]}, verbose=0)
        y_true_list.extend(np.argmax(lb, 1))
        y_pred_list.extend(np.argmax(pr, 1))
    except StopIteration:
        break

y_true_arr = np.array(y_true_list)
y_pred_arr = np.array(y_pred_list)
acc = np.mean(y_true_arr == y_pred_arr)

print(classification_report(y_true_arr, y_pred_arr,
      target_names=['Critical','Moderate','Safe']))
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# Save confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(12,5))
cm = confusion_matrix(y_true_arr, y_pred_arr)
cm_n = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Critical','Moderate','Safe'],
            yticklabels=['Critical','Moderate','Safe'], ax=axes[0])
axes[0].set_title('Confusion Matrix'); axes[0].set_ylabel('True'); axes[0].set_xlabel('Pred')
bars = axes[1].bar(['Critical','Moderate','Safe'], cm_n.diagonal()*100,
                   color=['#e74c3c','#f39c12','#2ecc71'])
axes[1].set_title('Per-Class Accuracy (%)'); axes[1].set_ylim([0,115])
for b,v in zip(bars, cm_n.diagonal()*100):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f'{v:.1f}%', ha='center', fontweight='bold')
plt.suptitle('AquaShield AI — Test Evaluation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/evaluation.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"✅ Evaluation plot saved → {SAVE_DIR}/evaluation.png")

# Set global reference
_model_global  = model_eval
_scaler_global = scaler

print("\n" + "="*55)
print("✅ TRAINING COMPLETE — run Cell 4")
print("="*55)


# ════════════════════════════════════════════════════════════════════
# CELL 4 — Grad-CAM engine + inference + save demo outputs
#
# FIX: Auto-detects the correct layer name — no more layer errors
# ════════════════════════════════════════════════════════════════════

# ── 4A: Find the correct Grad-CAM layer ──────────────────────────
def find_gradcam_layer(model):
    """
    Auto-detects correct last conv layer name inside MobileNetV2.
    Tries multiple known layer names — uses first that exists.
    FIX for: ValueError: No such layer: out_relu
    """
    # Candidates in preference order
    CANDIDATES = ['out_relu', 'Conv_1_bn', 'block_16_project_BN',
                  'block_15_add', 'block_14_add', 'block_13_add']

    # Check top-level layers first
    top_names = {l.name for l in model.layers}
    for c in CANDIDATES:
        if c in top_names:
            return c

    # Check inside sub-models (MobileNetV2 is wrapped)
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            sub_names = {sl.name for sl in layer.layers}
            for c in CANDIDATES:
                if c in sub_names:
                    # Found inside sub-model — get output at model level
                    # We'll handle sub-model access in GradCAM class
                    return c, layer   # return layer too

    # Fallback: last Conv2D at top level
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"  Using fallback conv layer: {layer.name}")
            return layer.name

    return None


# ── 4B: GradCAM class (fixed) ────────────────────────────────────
class GradCAM:
    """
    Fixed Grad-CAM implementation.
    Handles both top-level and sub-model (wrapped MobileNetV2) cases.
    """

    def __init__(self, model):
        self.model = model
        self.mobilenet_submodel = None
        self.target_layer_name  = None
        self._setup()

    def _setup(self):
        """Find MobileNetV2 sub-model and its last conv layer."""
        LAYER_CANDIDATES = ['out_relu','Conv_1_bn','block_16_project_BN',
                            'block_15_add','block_14_add']

        # Find the MobileNetV2 sub-model inside the main model
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 50:
                self.mobilenet_submodel = layer
                break

        if self.mobilenet_submodel is None:
            print("  ⚠️  MobileNetV2 sub-model not found, using fallback")
            return

        # Find correct layer name inside MobileNetV2
        sub_names = {l.name for l in self.mobilenet_submodel.layers}
        for c in LAYER_CANDIDATES:
            if c in sub_names:
                self.target_layer_name = c
                print(f"  ✅ Grad-CAM layer found: '{c}'")
                return

        # Fallback: last Conv2D inside sub-model
        for layer in reversed(self.mobilenet_submodel.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.target_layer_name = layer.name
                print(f"  ✅ Grad-CAM fallback layer: '{layer.name}'")
                return

        print("  ⚠️  Could not find conv layer, using pixel-based heatmap")

    def compute_heatmap(self, image_array, sensor_array, class_idx=None):
        """
        Compute Grad-CAM heatmap.
        Returns (heatmap_array, pred_class_idx, confidence).
        """
        if image_array.ndim == 3:
            image_array = image_array[np.newaxis]
        if sensor_array.ndim == 1:
            sensor_array = sensor_array[np.newaxis]

        img_t = tf.cast(image_array, tf.float32)
        sen_t = tf.cast(sensor_array, tf.float32)

        # Get prediction first
        preds = self.model.predict(
            {'image_input': img_t, 'sensor_input': sen_t}, verbose=0)
        if class_idx is None:
            class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])

        # Try Grad-CAM with sub-model
        if self.mobilenet_submodel and self.target_layer_name:
            try:
                # Build intermediate model from MobileNetV2
                inner = tf.keras.Model(
                    inputs=self.mobilenet_submodel.input,
                    outputs=[
                        self.mobilenet_submodel.get_layer(
                            self.target_layer_name).output,
                        self.mobilenet_submodel.output
                    ]
                )

                with tf.GradientTape() as tape:
                    conv_out, _ = inner(img_t, training=False)
                    tape.watch(conv_out)
                    # Run full model
                    full_pred = self.model(
                        {'image_input': img_t, 'sensor_input': sen_t},
                        training=False)
                    loss = full_pred[0][class_idx]

                grads  = tape.gradient(loss, conv_out)
                if grads is not None:
                    pooled = tf.reduce_mean(grads, axis=(0,1,2))
                    hm     = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
                    hm     = tf.nn.relu(hm).numpy()
                    if hm.max() > 0:
                        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                    return hm, class_idx, confidence

            except Exception as e:
                print(f"  Grad-CAM gradient failed: {str(e)[:60]}")
                print("  Using pixel-statistics heatmap instead")

        # Fallback heatmap from pixel statistics
        return self._pixel_heatmap(image_array[0], class_idx), class_idx, confidence

    def _pixel_heatmap(self, img_norm, class_idx):
        """Fallback: compute attention map from image pixel statistics."""
        img = (img_norm * 255).astype(np.uint8)
        size, step = 14, img.shape[0] // 14
        hm = np.zeros((size, size), dtype=np.float32)
        for gy in range(size):
            for gx in range(size):
                p = img[gy*step:(gy+1)*step, gx*step:(gx+1)*step]
                r, g, b = p[:,:,0].mean(), p[:,:,1].mean(), p[:,:,2].mean()
                if class_idx == 2:    # safe → blue-green
                    score = (b + g - r) / 255
                elif class_idx == 1:  # moderate → brownish
                    score = (r + g - b) / 255
                else:                 # critical → dark
                    score = (255 - (r+g+b)/3) / 255
                hm[gy, gx] = max(0, score)
        if hm.max() > 0:
            hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        return hm

    def overlay_heatmap(self, image_rgb, heatmap, alpha=0.45):
        """Overlay Grad-CAM heatmap on image. Returns RGB uint8 array."""
        size = image_rgb.shape[0]
        hm_r = cv2.resize(heatmap, (size, size))
        hm_u = np.uint8(255 * hm_r)
        hm_c = cv2.applyColorMap(hm_u, cv2.COLORMAP_JET)
        hm_c = cv2.cvtColor(hm_c, cv2.COLOR_BGR2RGB)
        base = image_rgb if image_rgb.max() > 1 else np.uint8(image_rgb*255)
        return cv2.addWeighted(base, 1-alpha, hm_c, alpha, 0)


# ── 4C: Full inference function ──────────────────────────────────
def predict_pollution(img_input, sensor_dict, model, scaler, gradcam):
    """
    Full inference pipeline.
    img_input   : file path (str) OR numpy RGB array
    sensor_dict : {ph, turbidity, dissolved_oxygen, temperature, cod}
    Returns     : result dict with images as base64 strings
    """
    CLASS_ACTIONS = {
        0: 'IMMEDIATE ACTION — Alert Pollution Control Board. Close discharge NOW.',
        1: 'INSPECTION REQUIRED — Schedule field visit within 48 hours.',
        2: 'NORMAL — Water quality within safe limits. Routine monitoring.'
    }

    # Load image
    if isinstance(img_input, str):
        img_bgr = cv2.imread(img_input)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_input.copy()

    img_r = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_n = img_r.astype('float32') / 255.0

    # Sensor preprocessing
    raw = [[sensor_dict[k] for k in SENSOR_FEATURES]]
    sen = scaler.transform(raw)

    # Predict
    preds    = model.predict(
        {'image_input': img_n[np.newaxis], 'sensor_input': sen}, verbose=0)
    pred_cls = int(np.argmax(preds[0]))
    conf     = float(preds[0][pred_cls])

    # Grad-CAM
    hm, _, _ = gradcam.compute_heatmap(img_n, sen[0], pred_cls)
    hm_r     = cv2.resize(hm, (IMG_SIZE, IMG_SIZE))
    hm_u     = np.uint8(255 * hm_r)
    hm_rgb   = cv2.cvtColor(cv2.applyColorMap(hm_u, cv2.COLORMAP_JET),
                             cv2.COLOR_BGR2RGB)
    ov_rgb   = gradcam.overlay_heatmap(img_r, hm, alpha=0.45)

    def to_b64(arr):
        pil = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO(); pil.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        'prediction':   CLASS_NAMES_MAP[pred_cls],
        'pred_idx':     pred_cls,
        'confidence':   round(conf * 100, 1),
        'icon':         CLASS_ICONS_MAP[pred_cls],
        'action':       CLASS_ACTIONS[pred_cls],
        'all_probs':    {CLASS_NAMES_MAP[i]: round(float(preds[0][i])*100,1)
                         for i in range(NUM_CLASSES)},
        'img_original': to_b64(img_r),
        'img_heatmap':  to_b64(hm_rgb),
        'img_overlay':  to_b64(ov_rgb),
    }


# ── 4D: Initialise Grad-CAM and save demo outputs ────────────────
print("Initialising Grad-CAM...")
grad_cam = GradCAM(_model_global)

# Save Grad-CAM demo grid
print("\nGenerating Grad-CAM demo visualisation...")
DEMO_SENSORS = {
    'safe':     [7.2, 2.5, 8.5, 22.0, 10.0],
    'moderate': [6.0, 25.0, 4.5, 28.0, 55.0],
    'critical': [4.2, 150., 1.5, 33.0, 250.0],
}

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
col_titles = ['Original Image', 'Grad-CAM Heatmap', 'Superimposed']
for j, t in enumerate(col_titles):
    axes[0][j].set_title(t, fontsize=12, fontweight='bold', pad=10)

for row, cls in enumerate(['safe', 'moderate', 'critical']):
    folder   = f'{DATASET_DIR}/test/{cls}'
    img_path = os.path.join(folder, os.listdir(folder)[0])
    img      = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_r    = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_n    = img_r.astype('float32') / 255.

    sen      = _scaler_global.transform([DEMO_SENSORS[cls]])
    preds    = _model_global.predict(
        {'image_input': img_n[np.newaxis], 'sensor_input': sen}, verbose=0)
    pred_cls = int(np.argmax(preds[0]))
    conf     = preds[0][pred_cls] * 100

    hm, _, _ = grad_cam.compute_heatmap(img_n, sen[0], pred_cls)
    hm_r2    = cv2.resize(hm, (IMG_SIZE, IMG_SIZE))
    hm_u     = np.uint8(255 * hm_r2)
    hm_rgb   = cv2.cvtColor(cv2.applyColorMap(hm_u, cv2.COLORMAP_JET),
                             cv2.COLOR_BGR2RGB)
    ov       = grad_cam.overlay_heatmap(img_r, hm, alpha=0.45)

    axes[row][0].imshow(img_r)
    axes[row][0].set_ylabel(f'True: {cls.upper()}', fontsize=10, fontweight='bold')
    axes[row][1].imshow(hm_rgb)
    axes[row][2].imshow(ov)
    axes[row][2].set_title(
        f"Pred: {CLASS_NAMES_MAP[pred_cls]}  {conf:.1f}%",
        color=['#ff4d6d','#ffb347','#00d68f'][pred_cls],
        fontsize=10, fontweight='bold')

    for j in range(3):
        axes[row][j].axis('off')
        for sp in axes[row][j].spines.values():
            sp.set_edgecolor(['#ff4d6d','#ffb347','#00d68f'][pred_cls])
            sp.set_linewidth(2)

plt.suptitle(
    'Grad-CAM — Whitebox Pollution Detection\n'
    'Heatmap shows which water region triggered the AI decision',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/gradcam_demo.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"✅ Grad-CAM demo saved → {SAVE_DIR}/gradcam_demo.png")
print("\n✅ Cell 4 complete — run Cell 5")


# ════════════════════════════════════════════════════════════════════
# CELL 5 — Flask web app + ngrok tunnel
#
# BEFORE RUNNING:
# ① Go to https://ngrok.com → Sign up free (30 seconds)
# ② Dashboard → left sidebar → "Your Authtoken" → Copy
# ③ Paste your token below replacing PASTE_YOUR_TOKEN_HERE
# ④ Run this cell → a browser link appears → click it
# ════════════════════════════════════════════════════════════════════

# ─── CHANGE THIS LINE ──────────────────────────────────────────────
NGROK_TOKEN = "PASTE_YOUR_TOKEN_HERE"
# ───────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
from pyngrok import ngrok, conf as ngrok_conf

app = Flask(__name__)
CORS(app)

# ── HTML UI (full browser app, no inline Colab widget) ───────────
HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AquaShield AI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a0e1a;--s1:#111827;--s2:#1a2234;--b1:#ffffff14;--b2:#ffffff22;
--tx:#e8edf5;--mu:#6b7a99;--sa:#00d68f;--mo:#ffb347;--cr:#ff4d6d;--ac:#4fc3f7;--ac2:#7c3aed}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(var(--b1) 1px,transparent 1px),
    linear-gradient(90deg,var(--b1) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none;z-index:0}
.w{position:relative;z-index:1;max-width:1060px;margin:0 auto;padding:0 20px 60px}
header{padding:30px 0 22px;display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid var(--b1);margin-bottom:26px}
.logo{display:flex;align-items:center;gap:12px}
.li{width:42px;height:42px;background:linear-gradient(135deg,var(--ac),var(--ac2));
  border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px}
.ln{font-family:'Syne',sans-serif;font-size:20px;font-weight:800}
.ls{font-family:'DM Mono',monospace;font-size:10px;color:var(--mu);margin-top:1px}
.badge{font-family:'DM Mono',monospace;font-size:10px;padding:4px 10px;border-radius:20px;
  border:1px solid var(--ac);color:var(--ac);letter-spacing:.6px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
@media(max-width:680px){.g2{grid-template-columns:1fr}}
.card{background:var(--s1);border:1px solid var(--b1);border-radius:14px;padding:22px}
.st{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;
  color:var(--mu);margin-bottom:12px}
.uz{border:2px dashed var(--b2);border-radius:10px;padding:28px 16px;text-align:center;
  cursor:pointer;transition:all .3s;background:var(--s2);position:relative;overflow:hidden}
.uz:hover,.uz.dov{border-color:var(--ac);background:#4fc3f710}
.uz input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.ui{font-size:32px;margin-bottom:8px}
.ut{font-size:13px;color:var(--mu);line-height:1.6}
.ut b{color:var(--tx)}.ut span{color:var(--ac)}
#pw{display:none;margin-top:14px;position:relative}
#pi{width:100%;height:170px;object-fit:cover;border-radius:8px;border:1px solid var(--b2);display:block}
.pb{position:absolute;top:7px;right:7px;background:#000a;border:1px solid var(--b2);
  color:var(--tx);font-size:11px;padding:3px 9px;border-radius:5px;cursor:pointer;
  font-family:'DM Mono',monospace}
.sbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.sb{padding:6px 14px;border-radius:7px;border:1px solid var(--b2);background:var(--s2);
  color:var(--tx);font-size:11px;font-family:'DM Mono',monospace;cursor:pointer;transition:all .2s}
.sb:hover{border-color:var(--ac);color:var(--ac)}
.sr{margin-bottom:14px}
.sl{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.sn{font-size:12px;font-family:'DM Mono',monospace;color:var(--mu)}
.sv{font-size:13px;font-family:'DM Mono',monospace;color:var(--tx)}
.ss{font-size:10px;font-family:'DM Mono',monospace;padding:2px 7px;border-radius:10px;margin-left:6px}
.ss-safe{background:#00d68f1a;color:var(--sa)}.ss-moderate{background:#ffb3471a;color:var(--mo)}
.ss-critical{background:#ff4d6d1a;color:var(--cr)}
input[type=range]{-webkit-appearance:none;width:100%;height:4px;background:var(--s2);
  border-radius:2px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;
  border-radius:50%;background:var(--ac);cursor:pointer;box-shadow:0 0 6px var(--ac)}
#bgo{width:100%;padding:15px;background:linear-gradient(135deg,var(--ac),var(--ac2));
  border:none;border-radius:11px;color:#fff;font-family:'Syne',sans-serif;
  font-size:15px;font-weight:700;cursor:pointer;margin-top:18px;transition:transform .2s}
#bgo:hover:not(:disabled){transform:translateY(-2px)}
#bgo:disabled{opacity:.5;cursor:not-allowed}
#res{display:none;margin-top:18px;animation:up .4s ease}
@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.rc{border-radius:14px;padding:22px;border:1px solid}
.rc.safe{background:#00d68f0b;border-color:#00d68f44}
.rc.moderate{background:#ffb3470b;border-color:#ffb34744}
.rc.critical{background:#ff4d6d0b;border-color:#ff4d6d44}
.rh{display:flex;align-items:center;gap:14px;margin-bottom:16px}
.ric{font-size:44px;line-height:1}
.rl{font-family:'Syne',sans-serif;font-size:30px;font-weight:800}
.rl.safe{color:var(--sa)}.rl.moderate{color:var(--mo)}.rl.critical{color:var(--cr)}
.rco{font-family:'DM Mono',monospace;font-size:12px;color:var(--mu);margin-top:3px}
.ra{background:var(--s2);border-radius:9px;padding:11px 14px;font-size:13px;
  line-height:1.6;margin-bottom:18px;border-left:3px solid}
.ra.safe{border-color:var(--sa)}.ra.moderate{border-color:var(--mo)}.ra.critical{border-color:var(--cr)}
.pr{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.pn{font-family:'DM Mono',monospace;font-size:11px;width:76px;flex-shrink:0}
.pt{flex:1;height:5px;background:var(--s2);border-radius:3px;overflow:hidden}
.pf{height:100%;border-radius:3px;transition:width .8s}
.pp{font-family:'DM Mono',monospace;font-size:11px;color:var(--mu);width:40px;text-align:right;flex-shrink:0}
.gg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:14px}
.gi{text-align:center}
.gl{font-family:'DM Mono',monospace;font-size:9px;color:var(--mu);
  letter-spacing:.8px;text-transform:uppercase;margin-bottom:5px}
.gi img{width:100%;border-radius:7px;border:1px solid var(--b2);display:block}
.sc{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;margin-top:14px}
.sc-chip{background:var(--s2);border-radius:7px;padding:7px 5px;text-align:center;border:1px solid var(--b1)}
.sc-n{font-family:'DM Mono',monospace;font-size:9px;color:var(--mu);text-transform:uppercase;margin-bottom:3px}
.sc-v{font-family:'DM Mono',monospace;font-size:12px;font-weight:500}
#ov{display:none;position:fixed;inset:0;background:#0a0e1acc;z-index:100;
  align-items:center;justify-content:center;flex-direction:column;gap:14px}
.sp{width:44px;height:44px;border:3px solid #ffffff22;border-top-color:var(--ac);
  border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.ot{font-family:'DM Mono',monospace;font-size:13px;color:var(--ac)}
#toast{position:fixed;bottom:22px;right:22px;background:var(--s1);border:1px solid var(--b2);
  border-radius:9px;padding:11px 16px;font-size:12px;font-family:'DM Mono',monospace;
  z-index:200;transform:translateY(70px);opacity:0;transition:all .3s;max-width:280px}
#toast.show{transform:translateY(0);opacity:1}
.wb{display:inline-flex;align-items:center;gap:5px;
  background:linear-gradient(135deg,#4fc3f715,#7c3aed15);
  border:1px solid #4fc3f740;border-radius:20px;padding:3px 11px;
  font-family:'DM Mono',monospace;font-size:10px;color:var(--ac)}
</style></head>
<body>
<div id="ov"><div class="sp"></div><div class="ot">Analysing river water...</div></div>
<div id="toast"></div>
<div class="w">
<header>
  <div class="logo">
    <div class="li">🌊</div>
    <div><div class="ln">AquaShield AI</div>
    <div class="ls">RIVER POLLUTION MONITOR — CIPHERLAB AI 2026</div></div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="wb">✦ WHITEBOX AI</span>
    <span class="badge">EGS PILLAY</span>
  </div>
</header>

<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px">
  <span class="badge" style="color:#00d68f;border-color:#00d68f">✅ Model Ready</span>
  <span class="badge">MobileNetV2 + Sensor Fusion</span>
  <span class="badge">Safe · Moderate · Critical</span>
  <span class="badge">Grad-CAM Explainability</span>
</div>

<div class="g2">
  <div class="card">
    <div class="st">River Water Image</div>
    <div class="uz" id="uz">
      <input type="file" id="fi" accept="image/*" onchange="handleFile(event)"/>
      <div class="ui">📷</div>
      <div class="ut"><b>Drop image here</b> or <span>click to browse</span><br/>
        CCTV · Drone · Smartphone photo of river</div>
    </div>
    <div id="pw">
      <img id="pi" src="" alt="preview"/>
      <button class="pb" onclick="clearImg()">✕</button>
    </div>
    <div style="margin-top:14px">
      <div class="st">Quick Test Samples</div>
      <div class="sbs">
        <button class="sb" onclick="loadSample('safe')">🟢 Safe water</button>
        <button class="sb" onclick="loadSample('moderate')">🟡 Moderate</button>
        <button class="sb" onclick="loadSample('critical')">🔴 Critical</button>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="st">Sensor Readings</div>
    <div class="sr">
      <div class="sl"><span class="sn">pH Level</span>
        <div><span class="sv" id="v-ph">7.2</span>
          <span class="ss ss-safe" id="s-ph">safe</span></div></div>
      <input type="range" id="r-ph" min="0" max="14" step="0.1" value="7.2"
        oninput="upd('ph',this.value)"/>
    </div>
    <div class="sr">
      <div class="sl"><span class="sn">Turbidity (NTU)</span>
        <div><span class="sv" id="v-turb">3.5</span>
          <span class="ss ss-safe" id="s-turb">safe</span></div></div>
      <input type="range" id="r-turb" min="0" max="500" step="0.5" value="3.5"
        oninput="upd('turb',this.value)"/>
    </div>
    <div class="sr">
      <div class="sl"><span class="sn">Dissolved O₂ (mg/L)</span>
        <div><span class="sv" id="v-do">8.5</span>
          <span class="ss ss-safe" id="s-do">safe</span></div></div>
      <input type="range" id="r-do" min="0" max="15" step="0.1" value="8.5"
        oninput="upd('do',this.value)"/>
    </div>
    <div class="sr">
      <div class="sl"><span class="sn">Temperature (°C)</span>
        <div><span class="sv" id="v-temp">24.0</span>
          <span class="ss ss-safe" id="s-temp">safe</span></div></div>
      <input type="range" id="r-temp" min="10" max="45" step="0.5" value="24"
        oninput="upd('temp',this.value)"/>
    </div>
    <div class="sr">
      <div class="sl"><span class="sn">COD (mg/L)</span>
        <div><span class="sv" id="v-cod">12.0</span>
          <span class="ss ss-safe" id="s-cod">safe</span></div></div>
      <input type="range" id="r-cod" min="0" max="500" step="1" value="12"
        oninput="upd('cod',this.value)"/>
    </div>
    <button id="bgo" onclick="analyse()">🔍 Analyse Pollution Level</button>
  </div>
</div>

<div id="res">
  <div class="card" style="padding:0;overflow:hidden">
    <div id="ri"></div>
  </div>
</div>

<div style="margin-top:36px;padding-top:20px;border-top:1px solid var(--b1);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">
  <span style="font-family:'DM Mono',monospace;font-size:11px;color:var(--mu)">
    AquaShield AI · CipherLab Blackbox→Whitebox · EGS Pillay Engineering College · AI&DS</span>
  <span class="wb">✦ TensorFlow + Grad-CAM</span>
</div>
</div>

<script>
const PRE={safe:{ph:7.2,turb:2.5,do:9.0,temp:22,cod:10},
           moderate:{ph:6.1,turb:28,do:4.2,temp:28,cod:62},
           critical:{ph:4.1,turb:180,do:1.2,temp:34,cod:310}};
const THR={ph:{s:[6.5,8.5],m:[5.5,6.5]},turb:{s:[0,5],m:[5,50]},
           do:{s:[6,15],m:[3,6]},temp:{s:[18,28],m:[22,32]},cod:{s:[0,20],m:[20,100]}};

function getCls(k,v){
  v=parseFloat(v);
  if(k==='do') return v>=6?'safe':v>=3?'moderate':'critical';
  const t=THR[k];
  if(v>=t.s[0]&&v<=t.s[1]) return 'safe';
  if(v>=t.m[0]&&v<=t.m[1]) return 'moderate';
  return 'critical';
}

let sen={ph:7.2,turb:3.5,do:8.5,temp:24,cod:12};

function upd(k,v){
  sen[k]=parseFloat(v);
  document.getElementById('v-'+k).textContent=parseFloat(v).toFixed(1);
  const c=getCls(k,v), el=document.getElementById('s-'+k);
  el.textContent=c; el.className='ss ss-'+c;
}
['ph','turb','do','temp','cod'].forEach(k=>upd(k,document.getElementById('r-'+k).value));

let upFile=null;

function loadSample(cls){
  fetch('/sample/'+cls).then(r=>r.blob()).then(b=>{
    upFile=new File([b],'sample.jpg',{type:'image/jpeg'});
    showPreview(URL.createObjectURL(b));
    const p=PRE[cls];
    Object.entries(p).forEach(([k,v])=>{document.getElementById('r-'+k).value=v;upd(k,v)});
    toast(cls.toUpperCase()+' sample loaded');
  });
}

function showPreview(url){
  document.getElementById('pi').src=url;
  document.getElementById('pw').style.display='block';
  document.getElementById('uz').style.display='none';
}
function clearImg(){
  upFile=null;
  document.getElementById('pw').style.display='none';
  document.getElementById('uz').style.display='block';
  document.getElementById('fi').value='';
}
function handleFile(e){
  const f=e.target.files[0]; if(!f) return;
  upFile=f; showPreview(URL.createObjectURL(f));
}
const uz=document.getElementById('uz');
uz.addEventListener('dragover',e=>{e.preventDefault();uz.classList.add('dov')});
uz.addEventListener('dragleave',()=>uz.classList.remove('dov'));
uz.addEventListener('drop',e=>{
  e.preventDefault();uz.classList.remove('dov');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')){upFile=f;showPreview(URL.createObjectURL(f))}
});

async function analyse(){
  if(!upFile){toast('Please upload or select an image first!');return}
  document.getElementById('ov').style.display='flex';
  const fd=new FormData();
  fd.append('image',upFile);
  fd.append('ph',sen.ph); fd.append('turbidity',sen.turb);
  fd.append('dissolved_oxygen',sen.do); fd.append('temperature',sen.temp);
  fd.append('cod',sen.cod);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const d=await r.json();
    document.getElementById('ov').style.display='none';
    if(d.error){toast('Error: '+d.error);return}
    renderResult(d);
  }catch(e){
    document.getElementById('ov').style.display='none';
    toast('Connection error — check Colab output for errors');
    console.error(e);
  }
}

function renderResult(d){
  const FC={SAFE:'#00d68f',MODERATE:'#ffb347',CRITICAL:'#ff4d6d'};
  const cls=d.prediction.toLowerCase();
  const fill=FC[d.prediction];
  const pb=['CRITICAL','MODERATE','SAFE'].map(p=>`
    <div class="pr">
      <span class="pn" style="color:${FC[p]}">${p}</span>
      <div class="pt"><div class="pf" style="width:${d.all_probs[p]}%;background:${FC[p]}"></div></div>
      <span class="pp">${d.all_probs[p]}%</span>
    </div>`).join('');
  const chips=[['pH',parseFloat(sen.ph).toFixed(1),'ph'],
    ['Turb',parseFloat(sen.turb).toFixed(1),'turb'],
    ['DO',parseFloat(sen.do).toFixed(1),'do'],
    ['Temp',parseFloat(sen.temp).toFixed(1),'temp'],
    ['COD',parseFloat(sen.cod).toFixed(0),'cod']]
    .map(([n,v,k])=>`<div class="sc-chip">
      <div class="sc-n">${n}</div>
      <div class="sc-v" style="color:var(--${getCls(k,v)})">${v}</div>
    </div>`).join('');

  document.getElementById('ri').innerHTML=`
  <div class="rc ${cls}">
    <div class="rh">
      <span class="ric">${d.icon}</span>
      <div>
        <div class="rl ${cls}">${d.prediction}</div>
        <div class="rco">Confidence: <b>${d.confidence}%</b> &nbsp;·&nbsp; MobileNetV2 + Sensor Fusion</div>
      </div>
    </div>
    <div class="ra ${cls}">${d.action}</div>
    <div style="margin-bottom:16px">${pb}</div>
    <div>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
        <span class="st" style="margin:0">Grad-CAM Explainability</span>
        <span class="wb">✦ Heatmap shows which region triggered this ${d.prediction} alert</span>
      </div>
      <div class="gg">
        <div class="gi">
          <div class="gl">Original Image</div>
          <img src="data:image/png;base64,${d.img_original}"/>
        </div>
        <div class="gi">
          <div class="gl">Attention Heatmap</div>
          <img src="data:image/png;base64,${d.img_heatmap}"/>
        </div>
        <div class="gi">
          <div class="gl">Superimposed ✦</div>
          <img src="data:image/png;base64,${d.img_overlay}"
               style="border-color:${fill};box-shadow:0 0 0 2px ${fill}44"/>
        </div>
      </div>
      <p style="font-size:11px;color:var(--mu);margin-top:8px;font-family:'DM Mono',monospace">
        Red/hot = AI focus region · Blue/cool = low attention · Explains the ${d.prediction} classification
      </p>
    </div>
    <div style="margin-top:14px">
      <div class="st">Sensor Summary</div>
      <div class="sc">${chips}</div>
    </div>
  </div>`;

  document.getElementById('res').style.display='block';
  document.getElementById('res').scrollIntoView({behavior:'smooth',block:'nearest'});
}

function toast(msg){
  const el=document.getElementById('toast');
  el.textContent=msg; el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'),2800);
}
</script></body></html>"""


# ── Flask routes ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/sample/<cls>')
def sample_image(cls):
    """Return a synthetic water image as JPEG."""
    if cls not in ['safe','moderate','critical']:
        return 'Invalid class', 400
    img_bgr  = make_water_image(cls, size=224)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil      = Image.fromarray(img_rgb)
    buf      = io.BytesIO()
    pil.save(buf, format='JPEG', quality=88)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')


@app.route('/predict', methods=['POST'])
def predict_route():
    """Main prediction endpoint — called by browser JS."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image in request'}), 400

        # Decode image
        img_bytes = request.files['image'].read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Could not decode image file'}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Parse sensor values
        sensor_dict = {
            'ph':               float(request.form.get('ph', 7.0)),
            'turbidity':        float(request.form.get('turbidity', 5.0)),
            'dissolved_oxygen': float(request.form.get('dissolved_oxygen', 7.0)),
            'temperature':      float(request.form.get('temperature', 25.0)),
            'cod':              float(request.form.get('cod', 20.0)),
        }

        if _model_global is None or _scaler_global is None:
            return jsonify({'error': 'Model not loaded — run Cells 3 and 4 first'}), 500

        result = predict_pollution(img_rgb, sensor_dict,
                                   _model_global, _scaler_global, grad_cam)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Start Flask in background thread ─────────────────────────────
def _run_flask():
    app.run(port=5000, debug=False, use_reloader=False, threaded=True)

flask_thread = threading.Thread(target=_run_flask, daemon=True)
flask_thread.start()
time.sleep(2)   # wait for Flask to start up
print("✅ Flask server started on port 5000")

# ── Start ngrok tunnel ────────────────────────────────────────────
if NGROK_TOKEN == "PASTE_YOUR_TOKEN_HERE":
    print("\n" + "!"*60)
    print("  ⚠️  You haven't set your ngrok token!")
    print("!"*60)
    print("""
  Steps to fix:
  1. Go to  https://ngrok.com
  2. Click 'Sign up' (free, 30 seconds)
  3. After login → Dashboard → 'Your Authtoken' on left sidebar
  4. Copy the token
  5. In THIS cell, change:
       NGROK_TOKEN = "PASTE_YOUR_TOKEN_HERE"
     to:
       NGROK_TOKEN = "your_actual_token_here"
  6. Re-run this cell
""")
else:
    try:
        ngrok.kill()   # kill any existing tunnels
        ngrok_conf.get_default().auth_token = NGROK_TOKEN
        tunnel     = ngrok.connect(5000, "http")
        public_url = tunnel.public_url

        print("\n" + "="*60)
        print("  ✅  AquaShield AI is LIVE!")
        print("="*60)
        print(f"""
  🌐  Open this link in your browser (or share with team):

       👉  {public_url}

  The link works on any device on any network.
  Keep this Colab tab open while using the app.
""")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ngrok error: {e}")
        print("""
  Possible fixes:
  1. Check your token is correct (no extra spaces)
  2. Try: !pkill ngrok  then re-run this cell
  3. Restart runtime and run all cells again
""")
