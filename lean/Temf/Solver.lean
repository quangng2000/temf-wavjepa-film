import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic
import Temf.Path

namespace Temf

def reverseStep (state delta velocity : ℝ) : ℝ := state - delta * velocity

theorem exact_average_reaches_endpoint (state endpoint delta : ℝ) (hd : delta ≠ 0) :
    reverseStep state delta ((state - endpoint) / delta) = endpoint := by
  unfold reverseStep
  field_simp
  ring

theorem endpoint_squared_error (state endpoint delta oracle prediction : ℝ)
    (correct : state - delta * oracle = endpoint) :
    (reverseStep state delta prediction - endpoint) ^ 2 =
      delta ^ 2 * (prediction - oracle) ^ 2 := by
  rw [← correct]
  unfold reverseStep
  ring

theorem vector_endpoint_squared_error {n : ℕ}
    (state endpoint oracle prediction : Fin n → ℝ) (delta : ℝ)
    (correct : ∀ i, state i - delta * oracle i = endpoint i) :
    (∑ i, (reverseStep (state i) delta (prediction i) - endpoint i) ^ 2) =
      delta ^ 2 * ∑ i, (prediction i - oracle i) ^ 2 := by
  simp_rw [endpoint_squared_error _ _ _ _ _ (correct _)]
  rw [Finset.mul_sum]

/-- The conversion is undefined as a velocity parameterization at t = 0. -/
theorem coefficient_velocity_loss (state prediction target t weight : ℝ) (ht : t ≠ 0) :
    weight * (((state - prediction) / t - (state - target) / t) ^ 2) =
      (weight / t ^ 2) * (prediction - target) ^ 2 := by
  field_simp
  ring

theorem full_jump (prior velocity : ℝ) : reverseStep prior 1 velocity = prior - velocity := by
  simp [reverseStep]

/-- Converting the full JVP velocity target does not usually give raw clean data. -/
theorem corrected_coefficient_target (clean prior t r jvp : ℝ) :
    path clean prior t - t * ((prior - clean) - (t - r) * jvp) =
      clean + t * (t - r) * jvp := by
  unfold path
  ring

end Temf
