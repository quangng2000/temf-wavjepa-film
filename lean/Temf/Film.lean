import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Tactic.Ring

namespace Temf

def film (feature scale shift : ℝ) : ℝ := (1 + scale) * feature + shift

theorem film_identity (feature : ℝ) : film feature 0 0 = feature := by simp [film]

/-- wavJEPA context and its adapters are held fixed along the flow-time JVP. -/
theorem film_time_derivative (feature : ℝ → ℝ) (t tangent scale shift : ℝ)
    (hf : HasDerivAt feature tangent t) :
    HasDerivAt (fun time => film (feature time) scale shift) ((1 + scale) * tangent) t := by
  exact (hf.const_mul (1 + scale)).add_const shift

theorem film_bounded_tangent (scale bound tangent : ℝ) (hs : |scale| ≤ bound) :
    |(1 + scale) * tangent| ≤ (1 + bound) * |tangent| := by
  rw [abs_mul]
  apply mul_le_mul_of_nonneg_right _ (abs_nonneg tangent)
  calc
    |1 + scale| ≤ |(1 : ℝ)| + |scale| := abs_add_le 1 scale
    _ ≤ 1 + bound := by simpa using add_le_add_left hs 1

theorem film_unbounded_scale (bound : ℝ) (hb : 0 ≤ bound) :
    |(1 + (bound - 1)) * (1 : ℝ)| = bound := by
  rw [show (1 + (bound - 1)) * (1 : ℝ) = bound by ring]
  exact abs_of_nonneg hb

/-- Identity initialization does not constrain scales after optimizer updates. -/
theorem film_can_erase_feature (feature : ℝ) : film feature (-1) 0 = 0 := by
  simp [film]

theorem film_energy_without_shift (feature scale : ℝ) :
    (film feature scale 0) ^ 2 = (1 + scale) ^ 2 * feature ^ 2 := by
  unfold film
  ring

/-- Fixed encoder features still permit nonzero adapter-parameter derivatives. -/
theorem film_adapter_derivative (scale shift : ℝ → ℝ) (theta ds db feature : ℝ)
    (hs : HasDerivAt scale ds theta) (hb : HasDerivAt shift db theta) :
    HasDerivAt (fun parameter => film feature (scale parameter) (shift parameter))
      (ds * feature + db) theta := by
  exact ((hs.const_add 1).mul_const feature).add hb

end Temf
