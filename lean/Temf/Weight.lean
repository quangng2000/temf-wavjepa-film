import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith

namespace Temf

/-- Released defaults: lambda1 = lambda2 = beta1 = 1; no mean normalization. -/
noncomputable def temporalWeight (delta : ℝ) : ℝ :=
  1 / ((1 + max delta (1 / 1000 : ℝ)) *
    Real.sqrt (1 + (max delta (1 / 1000 : ℝ)) ^ 2))

theorem temporal_weight_positive (delta : ℝ) : 0 < temporalWeight delta := by
  unfold temporalWeight
  have hd : 0 < max delta (1 / 1000 : ℝ) := lt_of_lt_of_le (by norm_num) (le_max_right _ _)
  positivity

theorem temporal_denominator_at_least_one (delta : ℝ) :
    1 ≤ (1 + max delta (1 / 1000 : ℝ)) *
      Real.sqrt (1 + (max delta (1 / 1000 : ℝ)) ^ 2) := by
  have hd : 0 ≤ max delta (1 / 1000 : ℝ) := le_trans (by norm_num) (le_max_right _ _)
  have hs := Real.sqrt_le_sqrt (show (1 : ℝ) ≤ 1 + (max delta (1 / 1000 : ℝ)) ^ 2 by nlinarith [sq_nonneg (max delta (1 / 1000 : ℝ))])
  rw [Real.sqrt_one] at hs
  nlinarith

theorem temporal_weight_at_most_one (delta : ℝ) : temporalWeight delta ≤ 1 := by
  unfold temporalWeight
  exact (div_le_one (by positivity)).mpr (temporal_denominator_at_least_one delta)

theorem weighted_error_nonnegative (delta error : ℝ) :
    0 ≤ temporalWeight delta * error ^ 2 := by
  exact mul_nonneg (le_of_lt (temporal_weight_positive delta)) (sq_nonneg error)

theorem zero_weighted_error_iff (delta error : ℝ) :
    temporalWeight delta * error ^ 2 = 0 ↔ error = 0 := by
  simp [ne_of_gt (temporal_weight_positive delta)]

end Temf
