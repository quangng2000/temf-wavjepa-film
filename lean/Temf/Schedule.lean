import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace Temf

noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- Progress is the original global step / total, not a short-test clock. -/
noncomputable def boundaryRatio (progress : ℝ) : ℝ :=
  9 / 10 - (8 / 10 : ℝ) * sigmoid (8 * (progress - 6 / 10))

theorem sigmoid_positive (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  positivity

theorem sigmoid_less_than_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  exact (div_lt_one (by positivity)).mpr (by linarith [Real.exp_pos (-x)])

theorem sigmoid_strict_mono : StrictMono sigmoid := by
  intro a b hab
  unfold sigmoid
  apply one_div_lt_one_div_of_lt (by positivity)
  simpa [add_comm] using add_lt_add_left (Real.exp_lt_exp.mpr (neg_lt_neg hab)) 1

theorem boundary_ratio_bounds (progress : ℝ) :
    1 / 10 < boundaryRatio progress ∧ boundaryRatio progress < 9 / 10 := by
  unfold boundaryRatio
  constructor <;> nlinarith [sigmoid_positive (8 * (progress - 6 / 10)),
    sigmoid_less_than_one (8 * (progress - 6 / 10))]

theorem boundary_ratio_decreases {p q : ℝ} (hpq : p < q) :
    boundaryRatio q < boundaryRatio p := by
  have h := sigmoid_strict_mono (show 8 * (p - 6 / 10) < 8 * (q - 6 / 10) by linarith)
  unfold boundaryRatio
  linarith

/-- Sorted logistic draws give valid intervals, but never an exact full jump. -/
theorem sorted_times (a b : ℝ) :
    0 < min (sigmoid a) (sigmoid b) ∧
    min (sigmoid a) (sigmoid b) ≤ max (sigmoid a) (sigmoid b) ∧
    max (sigmoid a) (sigmoid b) < 1 := by
  exact ⟨lt_min (sigmoid_positive a) (sigmoid_positive b),
    min_le_max, max_lt (sigmoid_less_than_one a) (sigmoid_less_than_one b)⟩

theorem mix_ratio_bounds (progress : ℝ) (hp : 0 ≤ progress) (hq : progress ≤ 1) :
    1 / 10 ≤ (3 / 10 : ℝ) - (2 / 10 : ℝ) * progress ∧
    (3 / 10 : ℝ) - (2 / 10 : ℝ) * progress ≤ 3 / 10 := by constructor <;> linarith

end Temf
