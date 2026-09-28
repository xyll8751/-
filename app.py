import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.feature import local_binary_pattern, hog
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import gradio as gr

# ==========================================
# 0. 基础辅助函数
# ==========================================
DATASET_DIR = r"D:\gongye\mvtec_ad"

def cv_imread_unicode(file_path):
    """支持中文/特殊路径的图片读取"""
    try:
        img_data = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取图片失败: {file_path}, Error: {e}")
        return None

def build_gabor_bank():
    filters = []
    for ksize in (9, 15):
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((ksize, ksize), 3.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filters.append(kernel)
    return filters

GABOR_FILTERS = build_gabor_bank()

# ==========================================
# 1. 方法 A（传统路线）：LBP + HOG + Gabor + OCSVM
# ==========================================
class AdvancedTraditionalDetector:
    def __init__(self, grid_size=(4, 4)):
        self.grid_size = grid_size
        self.scaler = StandardScaler()
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)

    def extract_features(self, img_gray):
        h, w = img_gray.shape
        gh, gw = h // self.grid_size[0], w // self.grid_size[1]
        feature_vector = []

        # 分块 LBP
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                patch = img_gray[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
                lbp = local_binary_pattern(patch, P=8, R=1, method="uniform")
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                feature_vector.extend(hist)

        # HOG
        hog_feat = hog(img_gray, orientations=8, pixels_per_cell=(32, 32),
                       cells_per_block=(1, 1), visualize=False)
        feature_vector.extend(hog_feat)

        # Gabor
        for kernel in GABOR_FILTERS:
            fimg = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
            feature_vector.append(np.mean(fimg))
            feature_vector.append(np.std(fimg))

        return np.array(feature_vector)

    def train(self, train_imgs_gray):
        feats = [self.extract_features(img) for img in train_imgs_gray]
        X = np.array(feats)
        X_scaled = self.scaler.fit_transform(X)
        self.ocsvm.fit(X_scaled)

    def predict(self, test_img_gray):
        start_time = time.time()
        feat = self.extract_features(test_img_gray).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        score = -self.ocsvm.decision_function(feat_scaled)[0]
        elapsed_time = (time.time() - start_time) * 1000
        return float(score), elapsed_time

# ==========================================
# 2. 方法 B（深度路线）：PaDiM (ResNet18)
# ==========================================
class EnhancedPaDiMDetector:
    def __init__(self, select_dim=100):
        resnet = models.resnet18(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5]).eval()
        self.layer2 = list(resnet.children())[5].eval()
        self.layer3 = list(resnet.children())[6].eval()
        
        self.select_dim = select_dim
        self.idx = None
        self.mean = None
        self.cov_inv = None
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, img_rgb):
        tensor = self.transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            f1 = self.layer1(tensor)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)

            f1_resized = nn.functional.interpolate(f1, size=(28, 28), mode='bilinear', align_corners=False)
            f3_resized = nn.functional.interpolate(f3, size=(28, 28), mode='bilinear', align_corners=False)

            merged = torch.cat([f1_resized, f2, f3_resized], dim=1).squeeze(0).numpy()
            return merged

    def train(self, train_imgs_rgb):
        feat_list = [self.extract_features(img) for img in train_imgs_rgb]
        features = np.array(feat_list)
        N, C, H, W = features.shape

        np.random.seed(42)
        self.idx = np.random.choice(C, self.select_dim, replace=False)
        features = features[:, self.idx, :, :]
        
        d = self.select_dim
        features = features.reshape(N, d, H * W)
        
        self.mean = np.mean(features, axis=0)
        self.cov_inv = np.zeros((H * W, d, d))
        
        identity = np.identity(d)
        for i in range(H * W):
            cov = np.cov(features[:, :, i], rowvar=False) + 0.01 * identity
            self.cov_inv[i] = np.linalg.inv(cov)

    def predict(self, test_img_rgb):
        start_time = time.time()
        feat = self.extract_features(test_img_rgb)[self.idx, :, :]
        d, H, W = feat.shape
        feat_flat = feat.reshape(d, H * W)

        dist_map = np.zeros(H * W)
        for i in range(H * W):
            delta = feat_flat[:, i] - self.mean[:, i]
            dist_map[i] = np.sqrt(np.dot(np.dot(delta, self.cov_inv[i]), delta.T))

        anomaly_map = dist_map.reshape(H, W)
        anomaly_map = cv2.resize(anomaly_map, (224, 224))
        anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), 0)
        
        image_score = float(np.max(anomaly_map))
        elapsed_time = (time.time() - start_time) * 1000
        return image_score, anomaly_map, elapsed_time

# 全局模型与类别状态
current_category = "bottle"
trad_detector = AdvancedTraditionalDetector()
deep_detector = EnhancedPaDiMDetector(select_dim=100)

# 获取 MVTec AD 目录下所有的子类别名称
def get_available_categories():
    if not os.path.exists(DATASET_DIR):
        return ["bottle"]
    cats = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return sorted(cats) if cats else ["bottle"]

# 加载数据与训练函数的封装
def load_and_train(category, max_samples=30):
    global current_category, trad_detector, deep_detector
    
    train_good_dir = os.path.join(DATASET_DIR, category, "train", "good")
    if not os.path.exists(train_good_dir):
        return f"❌ 找不到类别路径: {train_good_dir}"

    file_list = [f for f in os.listdir(train_good_dir) if f.endswith(('.png', '.jpg'))][:max_samples]
    if len(file_list) == 0:
        return f"❌ 目录 {train_good_dir} 中未找到图像文件"

    train_imgs_rgb, train_imgs_gray = [], []
    for fname in file_list:
        fpath = os.path.join(train_good_dir, fname)
        img_bgr = cv_imread_unicode(fpath)
        if img_bgr is None: continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))
        train_imgs_rgb.append(img_rgb)
        train_imgs_gray.append(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))

    # 重新拟合
    trad_detector.train(train_imgs_gray)
    deep_detector.train(train_imgs_rgb)
    current_category = category
    
    return f"✅ 成功加载类别 [{category}] 的 {len(train_imgs_rgb)} 张合格品并重新训练完成！"

# 默认初次训练
load_and_train("bottle", max_samples=30)

# ==========================================
# 3. Gradio 交互界面
# ==========================================
def process_inspection(input_img, trad_threshold, deep_threshold):
    if input_img is None:
        return None, "请上传测试图像"
    
    img_rgb = cv2.resize(input_img, (224, 224))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    score_trad, time_trad = trad_detector.predict(img_gray)
    score_deep, anomaly_map, time_deep = deep_detector.predict(img_rgb)
    
    # 深度热力图
    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-5)
    heatmap = cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    mask = (norm_map > 0.6).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

    status_trad = "❌ 异常 (缺陷)" if score_trad > trad_threshold else "✅ 正常 (合格)"
    status_deep = "❌ 异常 (缺陷)" if score_deep > deep_threshold else "✅ 正常 (合格)"

    report = f"""
    ### 🔬 性能与结果对比分析 (当前检测类别: **{current_category}**)

    | 检测路线 | 提取特征/模型 | 异常得分 | 判决阈值 | 判定结果 | 推理延时 (CPU) |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **传统路线** | LBP + HOG + Gabor + OCSVM | `{score_trad:.4f}` | `{trad_threshold:.2f}` | **{status_trad}** | `{time_trad:.2f} ms` |
    | **深度路线** | PaDiM (ResNet18多尺度) | `{score_deep:.4f}` | `{deep_threshold:.2f}` | **{status_deep}** | `{time_deep:.2f} ms` |
    """
    return overlay, report

available_categories = get_available_categories()

with gr.Blocks(title="工业无监督缺陷检测系统") as demo:
    gr.Markdown("# 🏭 工业纹理与结构缺陷无监督检测系统")
    
    with gr.Row():
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=available_categories, 
                value="bottle" if "bottle" in available_categories else available_categories[0], 
                label="选择训练类别 (MVTec AD)"
            )
            sample_slider = gr.Slider(minimum=10, maximum=100, value=30, step=5, label="训练合格品数量")
            retrain_btn = gr.Button("🔄 加载数据并重新训练模型", variant="primary")
            train_status = gr.Textbox(label="训练状态", value=f"✅ 当前模型已基于 [{current_category}] 训练完成", interactive=False)
            
            gr.Markdown("---")
            trad_thresh = gr.Slider(minimum=-1.0, maximum=2.0, value=0.1, step=0.05, label="传统路线阈值")
            deep_thresh = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.1, label="深度路线阈值")
            
        with gr.Column(scale=2):
            input_image = gr.Image(label="上传待检测工业样品图片")
            submit_btn = gr.Button("🔍 开始缺陷检测与定位", variant="secondary")
            output_image = gr.Image(label="缺陷精准定位 (PaDiM 热力图 & 目标框)")
            output_report = gr.Markdown()

    # 绑定事件
    retrain_btn.click(
        fn=load_and_train,
        inputs=[category_dropdown, sample_slider],
        outputs=[train_status]
    )
    
    submit_btn.click(
        fn=process_inspection,
        inputs=[input_image, trad_thresh, deep_thresh],
        outputs=[output_image, output_report]
    )

if __name__ == "__main__":
    demo.launch()