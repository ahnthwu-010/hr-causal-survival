# Does Employee Training Actually Reduce Attrition?
**Causal Inference + Survival Analysis · IBM HR Analytics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](LINK_STREAMLIT_CỦA_BẠN)

> Phần lớn phân tích HR chỉ hỏi: "Ai sẽ nghỉ việc?"  
> Dự án này hỏi câu khó hơn: **"Nếu tôi can thiệp, tôi thay đổi được gì?"**

---

## Phát hiện chính

**Training không có causal effect lên retention — đây là selection bias.**

| | Naive (chưa adjust) | Sau Propensity Score Matching |
|---|---|---|
| Log-rank p-value | **0.044** ✗ có vẻ significant | **0.570** ✗ không significant |
| Kết luận | Training giữ người | Training không giữ người |
| Lý do sai lệch | Công ty gửi training cho người vốn đã loyal | — |

Cái thật sự drive retention:

| Yếu tố | Hazard Ratio | Ý nghĩa |
|---|---|---|
| Overtime (Yes) | **2.13** | Làm thêm giờ tăng nguy cơ quit 113% — yếu tố số 1 |
| Marital: Single | **1.76** | Độc thân quit nhiều hơn 76% |
| Job Level | **0.79** | Mỗi bậc thăng tiến giảm 21% nguy cơ |
| Job Involvement | **0.78** | Engaged cao giảm 22% nguy cơ |
| Training (≥3×) | **0.94** | p=0.545 — không có ý nghĩa thống kê |

**Khuyến nghị business:** Ngân sách training đang chi sai chỗ. Tập trung vào kiểm soát overtime, lộ trình thăng tiến rõ ràng, và tăng job involvement — không phải tần suất training.

---

## Tại sao dự án này khác biệt

Phần lớn HR analytics dừng ở predictive: "Người này có 73% khả năng nghỉ." Dự án này đi xa hơn — **causal inference**: nếu công ty can thiệp bằng training, liệu tỷ lệ đó có thay đổi thật không?

Câu trả lời: Không. Và đây là insight đắt giá hơn con số 73% rất nhiều.

---

## Kiến trúc phân tích
1. EDA + Treatment Definition

└── TrainingTimesLastYear ≥ 3 = Treated (798 người)

└── < 3 = Control (672 người)
2. Naive Kaplan-Meier

└── p = 0.044 — trông có vẻ significant

└── ⚠️ CHƯA kiểm soát confounding
. Propensity Score Model

└── Logistic Regression trên 18 confounders

└── Accuracy 56.2% — lý tưởng cho PSM (không phải predictive)

└── Common Support: overlap hoàn hảo
4. Nearest Neighbor Matching 1:1

└── Caliper = 0.2 × SD(PS) — chuẩn Rosenbaum & Rubin 1985

└── 638 cặp matched (79.9% match rate)
5. Balance Check (Love Plot)

└── Mean SMD: 0.041 → 0.016 (giảm 61%)

└── Tất cả biến đạt SMD < 0.1 sau matching
6. Causal Survival Analysis

└── Cox PH trên matched sample

└── Concordance: 0.860

└── Schoenfeld residuals test
7. Bootstrap ATE với 95% CI

└── ATE(t) = S_trained(t) − S_control(t)

└── 500 iterations, paired bootstrap

└── CI chứa 0 tại mọi thời điểm → không có causal effect


---

## Kết quả model

| Model | Concordance | Ghi chú |
|---|---|---|
| Cox PH — Naive (toàn bộ data) | — | Biased, không dùng |
| Cox PH — Matched sample | **0.860** | Model cuối |

---

## Cài đặt và chạy

```bash
git clone https://github.com/ahnthwu-010/hr-causal-survival
cd hr-causal-survival
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Stack

- **Python** · pandas · numpy · matplotlib
- **lifelines** · Kaplan-Meier · Cox PH · Schoenfeld residuals  
- **scikit-learn** · Logistic Regression · StandardScaler
- **scipy** · Nearest Neighbor Matching
- **Streamlit** · deployment-ready app
- **Dataset**: [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## Tác giả

Sinh viên Thống kê K49 · Đại học Cần Thơ · Chuyên ngành Data Science

*Dự án này là phần 2 trong chuỗi portfolio DS — tập trung vào Causal Inference, kỹ năng phân biệt correlation với causation mà phần lớn DS tự học không có.*