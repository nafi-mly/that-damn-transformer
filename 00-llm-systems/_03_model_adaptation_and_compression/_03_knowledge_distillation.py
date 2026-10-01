import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

torch.manual_seed(42)


class TeacherModel(nn.Module):
    """
    Over-parameterized Teacher Model.
    Higher capacity to learn complex feature representations.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class StudentModel(nn.Module):
    """
    Compact Student Model.
    Significantly fewer parameters and shallower depth.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KnowledgeDistillationLoss(nn.Module):
    """
    Computes the combined Knowledge Distillation loss:
    L_total = alpha * L_soft + (1 - alpha) * L_hard

    where:
      L_soft = T^2 * KL_Divergence(Softmax(z_student / T), Softmax(z_teacher / T))
      L_hard = CrossEntropy(z_student, ground_truth_labels)
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha       = alpha
        self.ce_loss     = nn.CrossEntropyLoss()
        self.kl_div      = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor    
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_hard = self.ce_loss(student_logits, labels)

        soft_stud_prob_logs = F.log_softmax(student_logits / self.temperature, dim=-1) 
        soft_teacher_probs  = F.softmax(teacher_logits / self.temperature, dim=-1)

        loss_soft = self.kl_div(soft_stud_prob_logs, soft_teacher_probs) * (self.temperature**2)

        total_loss = (self.alpha * loss_hard) + ((1 - self.alpha) * loss_soft)

        return total_loss, loss_soft, loss_hard

def generate_synthetic_dataset(num_samples: int = 2000, input_dim: int = 128, num_classes: int = 10):
    """Generates synthetic classification dataset with structured non-linear decision boundaries."""
    X = torch.randn(num_samples, input_dim)
    # Project through fixed random weights to create non-trivial class assignment
    W_true = torch.randn(input_dim, num_classes)
    y_raw = torch.matmul(X, W_true) + 0.5 * torch.sin(X[:, :num_classes])
    y = torch.argmax(y_raw, dim=-1)
    return X, y

def train_epoch(model, loader, optimizer, criterion=None, teacher_model=None,device="cpu"):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        student_logits = model(X_batch)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X_batch)
            loss, _, _ = criterion(student_logits, teacher_logits, y_batch)
        else:
            loss = criterion(student_logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds    = torch.argmax(student_logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total   += X_batch.size(0)

    return total_loss / total, (correct / total) * 100.0

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = 0 
    total   = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits   = model(X_batch)
        preds    = torch.argmax(logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total   += X_batch.size(0)
    return (correct / total) * 100.0



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on device: {device}\n")

    # Hyperparameters
    INPUT_DIM = 128
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    EPOCHS = 100
    TEMPERATURE = 4.0
    ALPHA = 0.7

    # Load Data
    X_all, y_all = generate_synthetic_dataset(3600, INPUT_DIM, NUM_CLASSES)
    perm = torch.randperm(X_all.size(0))
    X_train, y_train = X_all[perm[:3000]], y_all[perm[:3000]]
    X_val,   y_val   = X_all[perm[3000:]], y_all[perm[3000:]]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate Models
    teacher = TeacherModel(INPUT_DIM, hidden_dim=256, num_classes=NUM_CLASSES).to(device)
    student_solo = StudentModel(INPUT_DIM, hidden_dim=32, num_classes=NUM_CLASSES).to(device)
    student_distilled = StudentModel(INPUT_DIM, hidden_dim=32, num_classes=NUM_CLASSES).to(device)

    # Ensure identical initialization between both students for fair comparison
    student_distilled.load_state_dict(student_solo.state_dict())

    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student_solo.parameters())

    print("--- MODEL PARAMETER COUNT ---")
    print(f"Teacher Model Parameters  : {teacher_params:,}")
    print(f"Student Model Parameters  : {student_params:,}")
    print(f"Compression Factor        : {teacher_params / student_params:.2f}x reduction\n")

    # -----------------------------------------------------------
    # STEP 1: Train the Teacher Model
    # -----------------------------------------------------------
    print("--- STAGE 1: Training Teacher Model (Hard Labels Only) ---")
    opt_teacher = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_epoch(teacher, train_loader, opt_teacher, criterion=ce_loss, device=device)
        if epoch % 5 == 0 or epoch == EPOCHS:
            val_acc = evaluate(teacher, val_loader, device=device)
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}%")

    teacher.eval()
    teacher_val_acc = evaluate(teacher, val_loader, device=device)
    print(f"[Teacher Final Accuracy]: {teacher_val_acc:.2f}%\n")

    # -----------------------------------------------------------
    # STEP 2: Train Student Solo (Hard Labels Only)
    # -----------------------------------------------------------
    print("--- STAGE 2: Training Solo Student (Hard Labels Only, No Teacher) ---")
    opt_student_solo = torch.optim.Adam(student_solo.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_epoch(student_solo, train_loader, opt_student_solo, criterion=ce_loss, device=device)
        if epoch % 5 == 0 or epoch == EPOCHS:
            val_acc = evaluate(student_solo, val_loader, device=device)
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}%")

    solo_val_acc = evaluate(student_solo, val_loader, device=device)
    print(f"[Solo Student Final Accuracy]: {solo_val_acc:.2f}%\n")

    # -----------------------------------------------------------
    # STEP 3: Train Student with Knowledge Distillation
    # -----------------------------------------------------------
    print(f"--- STAGE 3: Training Distilled Student (Temperature={TEMPERATURE}, Alpha={ALPHA}) ---")
    opt_student_distilled = torch.optim.Adam(student_distilled.parameters(), lr=1e-3)
    kd_loss_fn = KnowledgeDistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_epoch(
            student_distilled, 
            train_loader, 
            opt_student_distilled, 
            criterion=kd_loss_fn, 
            teacher_model=teacher, 
            device=device
        )
        if epoch % 5 == 0 or epoch == EPOCHS:
            val_acc = evaluate(student_distilled, val_loader, device=device)
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Combined Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}%")

    distilled_val_acc = evaluate(student_distilled, val_loader, device=device)
    print(f"[Distilled Student Final Accuracy]: {distilled_val_acc:.2f}%\n")

    # -----------------------------------------------------------
    # FINAL RESULTS SUMMARY
    # -----------------------------------------------------------
    print("=========================================================")
    print("               DISTILLATION RESULTS SUMMARY              ")
    print("=========================================================")
    print(f" Teacher Model Validation Accuracy      : {teacher_val_acc:.2f}%")
    print(f" Solo Student Validation Accuracy        : {solo_val_acc:.2f}%")
    print(f" Distilled Student Validation Accuracy   : {distilled_val_acc:.2f}%")
    print("---------------------------------------------------------")
    print(f" Accuracy Boost from Distillation       : +{distilled_val_acc - solo_val_acc:.2f}%")
    print(f" Performance Recovered relative to Teacher: {((distilled_val_acc - solo_val_acc) / (teacher_val_acc - solo_val_acc + 1e-8)) * 100:.1f}%")
    print("=========================================================")