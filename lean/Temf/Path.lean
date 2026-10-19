import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Tactic.Ring

namespace Temf

/-- One coefficient of a straight path; the prior is an explicit design choice. -/
def path (clean prior time : ℝ) : ℝ := (1 - time) * clean + time * prior

theorem path_clean (clean prior : ℝ) : path clean prior 0 = clean := by
  simp [path]

theorem path_prior (clean prior : ℝ) : path clean prior 1 = prior := by
  simp [path]

theorem path_derivative (clean prior time : ℝ) :
    HasDerivAt (path clean prior) (prior - clean) time := by
  unfold path
  convert (((hasDerivAt_id time).const_sub 1).mul_const clean).add
    ((hasDerivAt_id time).mul_const prior) using 1
  ring

theorem reverse_path_step (clean prior time destination : ℝ) :
    path clean prior time - (time - destination) * (prior - clean) =
      path clean prior destination := by
  unfold path
  ring

def noisyPath (clean damaged low high noise time : ℝ) : ℝ :=
  path clean damaged time + path low high time * noise

theorem noise_floor_changes_endpoint (clean damaged low high noise : ℝ) :
    noisyPath clean damaged low high noise 0 = clean + low * noise := by
  simp [noisyPath, path]

theorem noisy_prior_endpoint (clean damaged low high noise : ℝ) :
    noisyPath clean damaged low high noise 1 = damaged + high * noise := by
  simp [noisyPath, path]

theorem noisy_path_derivative (clean damaged low high noise time : ℝ) :
    HasDerivAt (noisyPath clean damaged low high noise)
      (damaged - clean + (high - low) * noise) time := by
  exact (path_derivative clean damaged time).add
    ((path_derivative low high time).mul_const noise)

end Temf
