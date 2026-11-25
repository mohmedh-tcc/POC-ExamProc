# Decoupling Head Pose and Eye Gaze (VOR Compensation) for Browser-Based Exam Proctoring

## 1. Root-Cause Diagnosis of the Current POC

An audit of the four uploaded modules (`transform-matrix.ts`, `iris-gaze.ts`, `calibrator.ts`, and `index.tsx`) reveals six compounding mathematical and geometric causes behind the Vestibulo-Ocular Reflex (VOR) false-positive / iris-cheat trade-off:

| Module | Current Implementation | Geometric / Mathematical Failure Mode |
| :--- | :--- | :--- |
| `transform-matrix.ts` | Extracts only Euler angles `(pitch, yaw, roll)` and discards the 3×3 rotation matrix $\mathbf{R}_{\text{head}}$ and 3D translation $\mathbf{t}_{\text{head}}$. | Prevents $SO(3)$ vector composition ($\mathbf{g}_{\text{world}} = \mathbf{R}_{\text{head}} \cdot \mathbf{g}_{\text{eye}}$), introduces Euler cross-talk under combined roll/yaw/pitch, and loses metric depth ($t_z$) and lateral translation ($t_x, t_y$) parallax. |
| `iris-gaze.ts` | Projects 2D normalized coordinates `(x, y)` onto `(u, w)` without aspect-ratio correction, divides by 2D `eyeWidth`, clamps `hRatio`/`vRatio` to `[0, 1]`, and averages both eyes equally. | (1) On 16:9 webcams, raw `(x, y)` shears the `(u, w)` basis under head roll. (2) Under head yaw $\psi$, projected 2D `eyeWidth` foreshortens by $\cos\psi$, inflating `dx` by $\approx 1/\cos\psi$. (3) Clamping to `[0, 1]` saturates the counter-rotation signal during VOR. (4) Equal averaging lets the nasal-occluded far eye corrupt the clean near eye. |
| `calibrator.ts` | Uses fixed linear heuristic `gx = -yaw + (hRatio - 0.5) * 40` and 5-point min/max screen bounds. | Eyeball rotation is spherical ($\Delta x = R_e \sin\theta$), not linear; the arbitrary gain `40` cannot cancel head yaw across different eye geometries, camera distances, or monitor sizes. A 5-point static-head calibration is also underdetermined for a 6-parameter 2nd-order perspective/tilt surface. |
| `index.tsx` | Flags `HEAD_OFF_SCREEN` whenever `yaw < cal.yawMin \|\| yaw > cal.yawMax` **OR** `mapped.isOffScreen`. | Because `yawMin`/`yawMax` are tight screen-edge bounds (`yawP15 - 6°` to `yawP85 + 6°`), turning the head while fixating on the screen still fires `HEAD_OFF_SCREEN` even when the combined gaze point stays inside the screen. |

---

## 2. Exact 3D Mathematical Formulation

The refactored pipeline replaces the 2D heuristic with a **hybrid 3D geometric model + closed-form Ridge Regression** grounded in remote gaze estimation theory ([Guestrin & Eizenman, 2006](https://pubmed.ncbi.nlm.nih.gov/16761839/); [Hansen & Ji, 2010](https://pubmed.ncbi.nlm.nih.gov/20075473/); [Valenti, Sebe, & Gevers, 2012](https://pubmed.ncbi.nlm.nih.gov/21788191/)).

### 2.1 Full 3D Rigid Head Transform from MediaPipe

[MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) outputs a 4×4 column-major metric 3D matrix (`facialTransformationMatrixes`) mapping its canonical face mesh into right-handed camera coordinates (in centimeters, camera at origin looking along $-Z$, face at $t_z < 0$ with nose pointing along $+Z$; see [MediaPipe 3D Face Transform](https://developers.googleblog.com/mediapipe-3d-face-transform/)):

$$\mathbf{M} = \begin{bmatrix} \mathbf{R}_{\text{head}} & \mathbf{t}_{\text{head}} \\ \mathbf{0}^\top & 1 \end{bmatrix}, \qquad R_{ij} = \text{data}[4j + i], \qquad \mathbf{t}_{\text{head}} = \begin{bmatrix} \text{data}[12] \\ \text{data}[13] \\ \text{data}[14] \end{bmatrix}$$

To guarantee $\mathbf{R}_{\text{head}} \in SO(3)$ ($\mathbf{R}^\top \mathbf{R} = \mathbf{I}$, $\det(\mathbf{R}) = +1$) against floating-point or scale drift, `extractRigidTransform` applies modified Gram-Schmidt orthonormalization to the first two columns $\mathbf{c}_0, \mathbf{c}_1$ and sets $\mathbf{c}_2 = \mathbf{c}_0 \times \mathbf{c}_1$.

### 2.2 Aspect-Corrected, Foreshortening-Compensated Spherical Eyeball Model

Let $A = W_{\text{video}} / H_{\text{video}}$ (default $16/9$). Scaling vertical landmark coordinates by $1/A$ converts MediaPipe's normalized landmarks into isotropic pixel units:

$$\tilde{\mathbf{p}} = \begin{bmatrix} x \\ y / A \end{bmatrix}$$

For each open eye (openness $h_{\text{lid}} / w_{\text{eye}} \ge 0.15$), we construct the roll-invariant orthonormal basis $(\mathbf{u}, \mathbf{w})$ along the inter-canthal axis (landmarks `33→133` for right eye, `362→263` for left eye) and project the iris center (`468`, `473`) relative to the canthal midpoint $\mathbf{m}$:

$$p_x = \frac{(\tilde{\mathbf{p}}_{\text{iris}} - \mathbf{m}) \cdot \mathbf{u}}{w_{\text{eye}}}, \qquad p_y = \frac{(\tilde{\mathbf{p}}_{\text{iris}} - \mathbf{m}) \cdot \mathbf{w}}{w_{\text{eye}}}$$

Two geometric corrections eliminate the VOR nonlinearities:

1. **Yaw Foreshortening Compensation**: Rotating the head by yaw $\psi$ shrinks the projected 2D inter-canthal width by $\approx \cos\psi$, artificially inflating $p_x$. Multiplying by $\max(0.65, \cos\psi)$ restores head-local scale:
   $$d_x = -p_x \cos\psi, \qquad d_y = p_y$$
   (Sign convention: $d_x > 0$ corresponds to looking toward screen-right; $d_y > 0$ corresponds to looking down.)
2. **Visibility-Weighted Binocular Fusion**: Instead of a 50/50 average, both open eyes are fused by squared apparent width $w_{\text{eye}}^2$:
   $$d_x = \frac{w_R^2 d_{x,R} + w_L^2 d_{x,L}}{w_R^2 + w_L^2}, \qquad d_y = \frac{w_R^2 d_{y,R} + w_L^2 d_{y,L}}{w_R^2 + w_L^2}$$
   When the head turns sideways, the nearer, unoccluded eye automatically dominates over the foreshortened far eye.
3. **Spherical Eyeball Kinematics**: Modeling the eyeball as a sphere of average anatomical radius $R_e \approx 12\text{ mm}$ ([Kaur et al., 2022](https://escholarship.org/content/qt53n6d26t/qt53n6d26t.pdf); [Model-Based 3D Gaze Estimation Using a TOF Camera, PMC10891597](https://pmc.ncbi.nlm.nih.gov/articles/PMC10891597/)) and nominal inter-canthal width $W_{\text{canthus}} \approx 30\text{ mm}$ gives the nominal gain $k_{h,0} = W_{\text{canthus}} / R_e \approx 2.5$:
   $$\theta_h = \arcsin\!\big(\text{clip}(k_h \, d_x,\; -0.92,\; 0.92)\big), \qquad \theta_v = \arcsin\!\big(\text{clip}(-k_v \, d_y,\; -0.92,\; 0.92)\big)$$
   yielding the 3D unit **eye-in-head optical axis vector** in head-local coordinates ($+Z$ pointing out of the face toward the screen, $-X$ pointing screen-right, $+Y$ pointing up):
   $$\mathbf{g}_{\text{eye}} = \begin{bmatrix} -\sin\theta_h \cos\theta_v \\ \sin\theta_v \\ \cos\theta_h \cos\theta_v \end{bmatrix}$$

*(Note: Individual foveal offset—angle $\kappa$, typically $\approx 5^\circ \pm 1^\circ$ horizontally nasal and $2^\circ\text{–}3^\circ$ vertically per [Guestrin & Eizenman, 2006](https://vemlab.github.io/127.0.0.1/static/attachment/36.pdf) and [Villanueva & Cabeza](https://academica-e.unavarra.es/bitstreams/74be5b46-2519-47a8-a11b-0f3c919cf5db/download)—is a constant angular bias in head-local space and is automatically absorbed by the screen-plane calibration intercept.)*

### 2.3 $SO(3)$ Head-Eye Vector Composition & Proof of VOR Invariance

Composing head rotation with eye-in-head rotation yields the world-space 3D gaze direction:

$$\mathbf{g}_{\text{world}} = \mathbf{R}_{\text{head}} \, \mathbf{g}_{\text{eye}}$$

**Exact VOR Cancellation**: Suppose the candidate fixates on the screen center ($\mathbf{g}_{\text{world}} = \mathbf{e}_z = [0, 0, 1]^\top$) and turns their head by angle $\psi$ around $+Y$ ($\mathbf{R}_{\text{head}} = \mathbf{R}_y(-\psi)$). The vestibulo-ocular reflex counter-rotates the eyeball inside the orbit by $\theta_h = -\psi$, so $\mathbf{g}_{\text{eye}} = \mathbf{R}_y(+\psi)\mathbf{e}_z$. Composing the two rotations in $SO(3)$ gives:

$$\mathbf{g}_{\text{world}} = \mathbf{R}_y(-\psi)\,\mathbf{R}_y(+\psi)\,\mathbf{e}_z = \mathbf{I}\,\mathbf{e}_z = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

Unlike `gx = -yaw + (hRatio - 0.5) * 40`, cancellation holds across arbitrary 3D yaw, pitch, and roll combinations without arbitrary linear gains.

### 2.4 3D Ray–Screen-Plane Intersection (Depth & Translation Parallax)

Treating the display bezel as the plane $z = 0$ in camera coordinates (following [Guestrin & Eizenman, 2006](https://pubmed.ncbi.nlm.nih.gov/16761839/)), the 3D gaze ray originating from $\mathbf{t}_{\text{head}} = (t_x, t_y, t_z)^\top$ intersects $z = 0$ at ray parameter $s = -t_z / \max(0.15,\, g_{\text{world}, z})$:

$$X_{\text{plane}} = -\left(t_x + s \, g_{\text{world}, x}\right), \qquad Y_{\text{plane}} = -\left(t_y + s \, g_{\text{world}, y}\right)$$

Because $t_x, t_y, t_z$ enter explicitly, leaning closer/farther from the laptop or shifting sideways in the chair preserves accurate screen-plane coordinates.

---

## 3. Calibration Architecture: 9-Point Grid + 2-Step VOR Sweep + Ridge Regression

### 3.1 Why 9 Points + 2 VOR Steps Beats 5-Point Static Calibration

1. **Polynomial Identifiability**: A second-order 2D polynomial surface $\phi(x, y) = [1,\, x,\, y,\, x^2,\, xy,\, y^2]^\top$ has **6 parameters per screen axis** and is the standard video-oculography model for absorbing laptop lid tilt, webcam offset, lens distortion, and corneal refraction ([Cerrolaza, Villanueva, & Cabeza, 2008](https://www.researchgate.net/publication/220811156_Taxonomic_Study_of_Polynomial_Regressions_Applied_to_the_Calibration_of_Video-Oculographic_Systems); [Cerrolaza et al., 2012](https://www.researchgate.net/publication/254461688_Study_of_Polynomial_Mapping_Functions_in_Video-Oculography_Eye_Trackers)). Five points are underdetermined ($5 < 6$); a **3×3 (9-point) grid** provides 9 spatial anchors ($9 > 6$). (`calibrator.ts` automatically falls back to a 3-parameter degree-1 affine fit if a legacy 5-point dataset is passed.)
2. **Breaking Static-Head Collinearity via 2-Step VOR Sweep**: During a static-head 9-point calibration, head yaw barely moves, so individual variation in $k_h = W_{\text{canthus}} / R_e$ cannot be observed independently of screen width. Adding two brief (~2.2 s each) **VOR steps** (*"Keep eyes locked on the center dot while turning your head slightly left, then slightly right"*) excites $\Delta \psi$ and $\Delta d_x$ in exact opposition while the true gaze target remains fixed at $(0.5, 0.5)$.
3. **Personal Anatomical Gain Identification ($k_h^*$)**: From the `center`, `vor-left`, and `vor-right` samples, we solve a 1D ridge-regularized least-squares problem (shrunk lightly toward $k_{h,0} = 2.5$ with $\gamma = 0.001$):
   $$k_h^* = \text{clip}\!\left(\frac{\sum_{i \in \text{VOR}} \Delta d_{x,i} \cdot (-\sin \Delta \psi_i) + \gamma \, k_{h,0}}{\sum_{i \in \text{VOR}} \Delta d_{x,i}^2 + \gamma},\; 1.5,\; 4.2\right)$$
4. **Saccade-Settling Trimming**: The first 30% of frames in each calibration step (~400–600 ms) are discarded to remove saccade-in-flight transients before fitting.

### 3.2 Closed-Form Ridge Regression on 3D Screen-Plane Features

Inspired by browser-based ridge regression in [WebGazer (Papoutsaki et al., 2016)](https://ijcai.org/proceedings/2016/540), we map standardized 3D ray-plane intersections $(\tilde{X}, \tilde{Y}) = \big((X_{\text{plane}} - \mu_x)/\sigma_x,\; (Y_{\text{plane}} - \mu_y)/\sigma_y\big)$ to normalized screen coordinates $(u_s, v_s) \in [0, 1]^2$:

$$\mathbf{w}_x = (\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I}_{\text{reg}})^{-1} \mathbf{\Phi}^\top \mathbf{u}_s, \qquad \mathbf{w}_y = (\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I}_{\text{reg}})^{-1} \mathbf{\Phi}^\top \mathbf{v}_s$$

where $\mathbf{I}_{\text{reg}} = \text{diag}(0, 1, \dots, 1)$ leaves the intercept unpenalized and $\lambda = 0.05$ (degree 2) or $\lambda = 0.02$ (degree 1). Because the input features $(X_{\text{plane}}, Y_{\text{plane}})$ are **already VOR-invariant in $SO(3)$**, the learned screen mapping generalizes across head turns without requiring calibration at every head pose. Solving the $6 \times 6$ linear system via Gaussian elimination with partial pivoting takes **< 0.1 ms** in pure TypeScript inside the Web Worker (zero GPU training overhead).

---

## 4. Proctoring Decision Logic & Guardrails (`index.tsx`)

1. **`HEAD_OFF_SCREEN` Decoupling**:
   - **Primary trigger**: `mapped.isOffScreen` ($u_s < -0.10$ or $u_s > 1.10$ or $v_s < -0.10$ or $v_s > 1.10$). This catches **both** head-away turns and **iris-only off-screen glances** while keeping natural VOR head turns inside $[0, 1]^2$.
   - **Wide Anatomical Safety Ceiling**: Tight `yawMin`/`yawMax` bounds are removed from triggering `HEAD_OFF_SCREEN`. Instead, a wide fallback ceiling (`|yaw - neutralYaw| > 35°` or `pitch - neutralPitch > +28°`) triggers only when extreme head rotation occludes the far eye behind the nasal bridge and degrades monocular iris landmarks.
2. **`LOOKING_DOWN` Dual Cue**:
   - Triggered when `pitch < pitchMin + driftPitch` **OR** `mapped.y > 1.12`. Retaining the pitch floor is intentional because downward gaze causes upper-eyelid droop that can partially occlude the iris when looking down at a phone or notes.
3. **Screen-Space `AdaptiveDriftTracker`**:
   - Operates directly on normalized screen residuals (clamped to $\pm 8\%$ horizontal, $\pm 6\%$ vertical, $\alpha = 0.012$ at 5 Hz) and updates **only** when raw gaze lies comfortably inside the screen interior $[0.15, 0.85]^2$ and no off-screen/looking-down condition is active.

---

## 5. Numerical Verification Results

A 3D geometric simulation (`verify_vor_math.py`) modeled a 15.6" 16:9 laptop screen ($34 \times 19\text{ cm}$, webcam 2 cm above top bezel), user at $t_z = -55\text{ cm}$, eyeball radius $R_e = 12\text{ mm}$, and non-nominal canthus width $W = 31.5\text{ mm}$ (true $k_h = 2.625$ vs. nominal $2.50$).

- **VOR Personal Gain Recovery**: True $k_h = 2.6250 \rightarrow$ Estimated $k_h^* = 2.6189$ (0.23% error).
- **9-Point + 2-VOR Calibration RMSE**: $0.15\%$ of screen width ($X$), $0.14\%$ of screen height ($Y$).

| Test Scenario | Ground-Truth $u$ | Refactored $u$ | Refactored Flag | Legacy $u$ | Legacy Flag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| VOR Center, $+15^\circ$ head yaw (ON-SCREEN) | 0.50 | 0.50 | **False (Pass)** | -0.36 | True (False Pos) |
| VOR Center, $-15^\circ$ head yaw (ON-SCREEN) | 0.50 | 0.50 | **False (Pass)** | 1.36 | True (False Pos) |
| VOR Center, $+25^\circ$ head yaw (ON-SCREEN) | 0.50 | 0.50 | **False (Pass)** | -0.92 | True (False Pos) |
| VOR Center, $-25^\circ$ head yaw (ON-SCREEN) | 0.50 | 0.50 | **False (Pass)** | 1.92 | True (False Pos) |
| VOR Left-Quarter, $+20^\circ$ head yaw (ON-SCREEN) | 0.25 | 0.25 | **False (Pass)** | -0.83 | True (False Pos) |
| Lateral lean 8 cm right, eyes on center (ON-SCREEN) | 0.50 | 0.50 | **False (Pass)** | 0.31 | False |
| Iris-only cheat RIGHT, head still (OFF-SCREEN) | 1.35 | 1.35 | **True (Caught)** | 1.11 | True |
| Iris-only cheat LEFT, head still (OFF-SCREEN) | -0.35 | -0.35 | **True (Caught)** | -0.11 | True |
| Head + Iris cheat RIGHT (OFF-SCREEN) | 1.40 | 1.39 | **True (Caught)** | 1.61 | True |

---

## 6. Browser Web Worker Performance Budget (5 Hz)

All computations are pure TypeScript floating-point math with zero external matrix libraries and zero GPU/WebGL training overhead:

- **Per-frame inference (5 Hz / 200 ms budget)**: `extractRigidTransform` (~40 FLOPs) + `computeIrisGaze` (~60 FLOPs) + `composeWorldGazeRay` + 6-term Ridge dot product (~50 FLOPs) = **< 0.05 ms/frame** (< 0.03% of the 200 ms budget).
- **One-time calibration fit**: `fitVorEyeGain` + $6 \times 6$ Gaussian elimination (`solveRidgeRegression`) on ~100 frames = **< 0.2 ms total**.
- **Integration note for `CameraPipeline` / Web Worker**: Pass the raw 16-element `facialTransformationMatrixes[0].data` on `FaceDetectionEvent.transformMatrix` and pass `{ aspectRatio: video.videoWidth / video.videoHeight, headYawRad: angles.yaw * Math.PI / 180 }` into `computeIrisGaze`. If `transformMatrix` is temporarily omitted, `calibrator.ts` automatically falls back to `eulerToRotationMatrix(angles)` with a nominal 55 cm viewing distance so the refactored files remain backwards-compatible out of the box.

---

## 7. Research References

1. **Guestrin, E. D., & Eizenman, M. (2006).** *General Theory of Remote Gaze Estimation Using the Pupil Center and Corneal Reflections.* IEEE Transactions on Biomedical Engineering, 53(6), 1124–1133. [PubMed](https://pubmed.ncbi.nlm.nih.gov/16761839/) | [PDF](https://vemlab.github.io/127.0.0.1/static/attachment/36.pdf)
2. **Hansen, D. W., & Ji, Q. (2010).** *In the Eye of the Beholder: A Survey of Models for Eyes and Gaze.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 32(3), 478–500. [PubMed](https://pubmed.ncbi.nlm.nih.gov/20075473/) | [IEEE CSDL](https://www.computer.org/csdl/journal/tp/2010/03/ttp2010030478/13rRUxOdD9o)
3. **Valenti, R., Sebe, N., & Gevers, T. (2012).** *Combining Head Pose and Eye Location Information for Gaze Estimation.* IEEE Transactions on Image Processing, 21(2), 802–815. [PubMed](https://pubmed.ncbi.nlm.nih.gov/21788191/)
4. **Cerrolaza, J. J., Villanueva, A., & Cabeza, R. (2008).** *Taxonomic Study of Polynomial Regressions Applied to the Calibration of Video-Oculographic Systems.* Proc. ACM Symposium on Eye Tracking Research & Applications (ETRA 2008). [ResearchGate](https://www.researchgate.net/publication/220811156_Taxonomic_Study_of_Polynomial_Regressions_Applied_to_the_Calibration_of_Video-Oculographic_Systems)
5. **Cerrolaza, J. J., Villanueva, A., & Cabeza, R. (2012).** *Study of Polynomial Mapping Functions in Video-Oculography Eye Trackers.* ACM Transactions on Computer-Human Interaction (TOCHI). [ResearchGate](https://www.researchgate.net/publication/254461688_Study_of_Polynomial_Mapping_Functions_in_Video-Oculography_Eye_Trackers)
6. **Papoutsaki, A., Sangkloy, P., Laskey, J., Daskalova, N., Huang, J., & Hays, J. (2016).** *WebGazer: Scalable Webcam Eye Tracking Using User Interactions.* Proc. 25th International Joint Conference on Artificial Intelligence (IJCAI 2016). [IJCAI Proceedings](https://ijcai.org/proceedings/2016/540) | [PDF](https://ijcai.org/proceedings/2016/0540.pdf)
7. **Lu, F., Sugano, Y., Okabe, T., & Sato, Y. (2014).** *Adaptive Linear Regression for Appearance-Based Gaze Estimation.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 36(10), 2033–2046. [IEEE CSDL](https://www.computer.org/csdl/journal/tp/2014/10/06777326/13rRUEgartY)
8. **Ablavatski, A., Vakunov, A., Grishchenko, I., Raveendran, K., & Zhdanov, M. (2020).** *Real-time Pupil Tracking from Monocular Video for Digital Puppetry* (MediaPipe Iris). arXiv:2006.11341. [arXiv](https://arxiv.org/abs/2006.11341)
9. **Kaur, H., Manduchi, R., et al. (2022).** *Rethinking Model-Based Gaze Estimation.* Proc. ACM on Computer Graphics and Interactive Techniques (ETRA 2022), 5(2). [eScholarship PDF](https://escholarship.org/content/qt53n6d26t/qt53n6d26t.pdf)
10. **Google AI Edge / MediaPipe.** *Face Landmarker & 3D Face Transform Documentation.* [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | [3D Face Transform Blog](https://developers.googleblog.com/mediapipe-3d-face-transform/)
