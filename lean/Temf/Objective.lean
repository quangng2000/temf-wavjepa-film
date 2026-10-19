import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace Temf

/-- A detached target and temporal weight are constants during this update. -/
theorem detached_loss_derivative (prediction target weight : ℝ) :
    HasDerivAt (fun p => weight * (p - target) ^ 2)
      (2 * weight * (prediction - target)) prediction := by
  convert (((hasDerivAt_id prediction).sub_const target).pow 2).const_mul weight using 1
  simp only [id_eq, Nat.cast_ofNat, Nat.reduceSub, pow_one, mul_one]
  ring

theorem positive_weight_preserves_gradient_sign (weight error : ℝ) (hw : 0 < weight) :
    0 < 2 * weight * error ↔ 0 < error := by
  exact mul_pos_iff_of_pos_left (by linarith)

/-- Any fixed positive weight admits arbitrarily large corrected-target gradients. -/
theorem weighting_does_not_bound_gradient (weight bound : ℝ) (hw : 0 < weight) :
    ∃ target : ℝ, bound < 2 * weight * (0 - target) := by
  refine ⟨-(bound + 1) / (2 * weight), ?_⟩
  have he : 2 * weight * (0 - (-(bound + 1) / (2 * weight))) = bound + 1 := by
    field_simp
    ring
  rw [he]
  linarith

/-- A synthetic corrected target of 1 can improve while error to oracle 0 grows. -/
theorem corrected_loss_is_not_endpoint_error :
    (((1 / 2 : ℝ) - 1) ^ 2 < (0 - 1 : ℝ) ^ 2) ∧
    ((0 - 0 : ℝ) ^ 2 < ((1 / 2 : ℝ) - 0) ^ 2) := by norm_num

/-- Prediction/target error alone omits the network's parameter sensitivity. -/
theorem parameter_loss_derivative (predict : ℝ → ℝ) (theta tangent target weight : ℝ)
    (hp : HasDerivAt predict tangent theta) :
    HasDerivAt (fun parameter => weight * (predict parameter - target) ^ 2)
      (2 * weight * (predict theta - target) * tangent) theta := by
  convert ((hp.sub_const target).pow 2).const_mul weight using 1
  ring

end Temf
