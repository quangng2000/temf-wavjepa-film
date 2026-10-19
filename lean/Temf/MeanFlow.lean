import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Tactic.Linarith

namespace Temf

/-- Differentiate displacement, not an assumed MeanFlow target identity. -/
theorem mean_flow_identity (trajectory average : ℝ → ℝ) (r t velocity jvp : ℝ)
    (displacement : ∀ s, (s - r) * average s = trajectory s - trajectory r)
    (hx : HasDerivAt trajectory velocity t) (hu : HasDerivAt average jvp t) :
    average t = velocity - (t - r) * jvp := by
  have product := ((hasDerivAt_id t).sub_const r).mul hu
  change HasDerivAt (fun s => (s - r) * average s)
    (1 * average t + (t - r) * jvp) t at product
  have equal_functions := funext displacement
  rw [equal_functions] at product
  have equal_derivatives := product.unique (hx.sub_const (trajectory r))
  linarith

/-- Destination and speech context are fixed in the partially applied predictor. -/
theorem state_time_jvp {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (predict : E × ℝ → ℝ) (linear : (E × ℝ) →L[ℝ] ℝ)
    (state : ℝ → E) (t : ℝ) (velocity : E)
    (hp : HasFDerivAt predict linear (state t, t))
    (hx : HasDerivAt state velocity t) :
    HasDerivAt (fun s => predict (state s, s)) (linear (velocity, 1)) t := by
  exact HasFDerivAt.comp_hasDerivAt (f := fun s => (state s, s)) t hp
    (hx.prodMk (hasDerivAt_id t))

theorem diagonal_target (velocity jvp t : ℝ) :
    velocity - (t - t) * jvp = velocity := by simp

/-- A mixed tangent equals the analytic tangent only under an extra condition. -/
theorem mixed_tangent_error (mix analytic predicted : ℝ) :
    mix * analytic + (1 - mix) * predicted - analytic =
      (1 - mix) * (predicted - analytic) := by ring

end Temf
