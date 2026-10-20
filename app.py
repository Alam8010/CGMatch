import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ── WideResNet-28-2 defined inline (no USB import needed) ─────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop  = nn.Dropout(p=drop_rate)
        self.shortcut = (stride != 1 or in_planes != out_planes)
        if self.shortcut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.convShortcut(out) if self.shortcut else x
        out = self.conv1(out)
        out = self.conv2(self.drop(self.relu2(self.bn2(out))))
        return out + shortcut

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0, num_classes=10):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) // 6
        self.conv1  = nn.Conv2d(3, n_channels[0], 3, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], BasicBlock, 2, drop_rate)
        self.bn1    = nn.BatchNorm2d(n_channels[3])
        self.relu   = nn.ReLU(inplace=True)
        self.fc     = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels)
        return self.fc(out)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2471, 0.2435, 0.2616)

CHECKPOINT = 'model_best.pth'

# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not __import__('os').path.exists(CHECKPOINT):
        return None

    model = WideResNet(depth=28, widen_factor=2, num_classes=10)
    ckpt  = torch.load(CHECKPOINT, map_location='cpu')

    if isinstance(ckpt, dict):
        if 'ema_model' in ckpt and ckpt['ema_model'] is not None:
            state_dict = ckpt['ema_model']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean = {k.replace('module.', '').replace('backbone.', ''): v
             for k, v in state_dict.items()
             if not k.startswith('fc_') and 'classifier' not in k}
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

def predict(model, img):
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]
    return probs.numpy()

def cgmatch_category(max_prob):
    if max_prob >= 0.90:
        return "Easy", "#22c55e", (
            "High confidence — during training, CGMatch would treat this as an "
            "**easy** sample: hard pseudo-label, standard cross-entropy loss."
        )
    elif max_prob >= 0.60:
        return "Ambiguous", "#f59e0b", (
            "Moderate confidence — CGMatch would treat this as **ambiguous**: "
            "soft label, GCE loss (more forgiving of noisy predictions)."
        )
    else:
        return "Hard", "#ef4444", (
            "Low confidence — CGMatch would **skip** this sample during training "
            "to avoid learning from an unreliable pseudo-label."
        )

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CGMatch Demo", page_icon="🔍", layout="centered")

st.title("CGMatch — CIFAR-10 Classifier")
st.caption(
    "Reproduction of *CGMatch: A Different Perspective of Semi-supervised Learning* "
    "(CVPR 2025) · IBA Karachi, Intro to ML"
)

with st.expander("⚠️ This model only knows 10 categories — click to see them", expanded=False):
    st.markdown(
        "The model was trained exclusively on **CIFAR-10**, which contains only:\n\n"
        "✈️ airplane &nbsp;|&nbsp; 🚗 automobile &nbsp;|&nbsp; 🐦 bird &nbsp;|&nbsp; "
        "🐱 cat &nbsp;|&nbsp; 🦌 deer &nbsp;|&nbsp; 🐶 dog &nbsp;|&nbsp; "
        "🐸 frog &nbsp;|&nbsp; 🐴 horse &nbsp;|&nbsp; 🚢 ship &nbsp;|&nbsp; 🚛 truck\n\n"
        "Anything else (a face, food, furniture…) will be force-fit to the nearest "
        "class. Low confidence in that case is expected and correctly flagged as **hard**."
    )

model = load_model()

if model is None:
    st.error(f"Checkpoint `{CHECKPOINT}` not found. Place it in the same folder as `app.py` and restart.")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.markdown("---")
    st.markdown(
        "Upload any image. The model predicts one of the 10 CIFAR-10 classes and "
        "shows how CGMatch would categorise this sample during training."
    )
    st.stop()

img   = Image.open(uploaded).convert("RGB")
probs = predict(model, img)

top3_idx  = probs.argsort()[::-1][:3]
max_prob  = float(probs.max())
top_class = CLASSES[int(probs.argmax())]

col_img, col_results = st.columns([1, 2])

with col_img:
    st.image(img, use_column_width=True)
    st.caption("Input (resized to 32×32 internally)")

with col_results:
    st.subheader(f"**{top_class.upper()}**")
    category, color, explanation = cgmatch_category(max_prob)
    st.markdown(
        f"<div style='background:{color}18;border-left:4px solid {color};"
        f"padding:10px 14px;border-radius:6px;margin-bottom:14px;'>"
        f"<span style='color:{color};font-weight:700;font-size:1.05em'>{category} sample</span>"
        f"<span style='color:#666;'> · {max_prob:.1%} confidence</span><br>"
        f"<small style='color:#444;'>{explanation}</small></div>",
        unsafe_allow_html=True
    )
    if max_prob < 0.40:
        st.warning(
            "Very low confidence — this image likely doesn't resemble any CIFAR-10 category. "
            "The prediction below is a best-guess among the 10 known classes."
        )
    st.markdown("**Top-3 predictions:**")
    for i in top3_idx:
        st.progress(float(probs[i]), text=f"{CLASSES[i]:<13} {probs[i]:.1%}")

st.markdown("---")
st.caption("Model: WideResNet-28-2 · CGMatch · CIFAR-10 40-label · Akash Alam & Mubeen Udin · IBA Karachi")
